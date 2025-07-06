import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTModel
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file
import editdistance

# ============ Dataset ============
class DocVQADataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row["imageid"])
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "question": row["question"],
            "answers": row["answers"].split("|||")
        }

# ============ Filtered Dataset Wrapper ============
class FilteredDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.valid_indices = []
        for i in range(len(base_dataset)):
            try:
                sample = base_dataset[i]
                if all(k in sample for k in ['image', 'question', 'answers']):
                    self.valid_indices.append(i)
            except:
                continue

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]

# ============ Collate Function ============
def collate_fn(batch):
    valid_batch = [item for item in batch if all(k in item for k in ["image", "question", "answers"])]
    pixel_values = torch.stack([item["image"] for item in valid_batch])
    questions = [item["question"] for item in valid_batch]
    answers = [item["answers"] for item in valid_batch]  # List[List[str]]
    return {
        "pixel_values": pixel_values,
        "questions": questions,
        "answers": answers
    }

# ============ ANLS ============
def normalized_levenshtein(pred, gt):
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    dist = editdistance.eval(pred, gt)
    norm = dist / max(len(pred), len(gt))
    return 1 - norm

def compute_anls(preds, gts_batch, threshold=0.5):
    scores = []
    for pred, gts in zip(preds, gts_batch):
        max_sim = max([normalized_levenshtein(pred, gt) for gt in gts])
        scores.append(max_sim if max_sim >= threshold else 0.0)
    return np.mean(scores)

# ============ Model Wrapper ============
class VisionLanguageModel(nn.Module):
    def __init__(self, vit_model, phi_model, projector, tokenizer):
        super().__init__()
        self.vit = vit_model
        self.phi = phi_model
        self.projector = projector
        self.tokenizer = tokenizer

    def forward(self, pixel_values, questions, answers=None):
        with torch.no_grad():
            image_feats = self.vit(pixel_values).last_hidden_state[:, 0, :]
        proj_feats = self.projector(image_feats).unsqueeze(1).to(torch.float16)

        prompts = [f"<image> Question: {q.strip()} Answer:" for q in questions]
        full_texts = [f"{p} {a[0].strip()}" for p, a in zip(prompts, answers)]

        tokenized = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized.input_ids.to(proj_feats.device)
        attention_mask = tokenized.attention_mask.to(proj_feats.device)

        input_embeds = self.phi.get_input_embeddings()(input_ids)
        input_embeds = torch.cat([proj_feats, input_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=proj_feats.device),
            attention_mask
        ], dim=1)

        labels = input_ids.clone()
        for i in range(len(labels)):
            prompt_len = len(self.tokenizer(prompts[i], return_tensors="pt").input_ids[0])
            labels[i, :prompt_len] = -100
        labels = torch.cat([
            torch.full((labels.shape[0], 1), fill_value=-100, dtype=torch.long, device=labels.device),
            labels
        ], dim=1)

        return self.phi(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        ), prompts

# ============ Training ============
def train_model(model, train_loader, val_loader, tokenizer, device, epochs=5, lr=1e-4):
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_anls = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            (outputs, _) = model(pixel_values, questions, answers)
            loss = outputs.loss

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        val_anls = evaluate_model(model, val_loader, tokenizer, device)
        if val_anls > best_anls:
            best_anls = val_anls
            save_file(model.projector.state_dict(), "best_projector.safetensors")
            print("✅ New best projector saved!")

# ============ Evaluation ============
def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Eval]"):
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            with torch.no_grad():
                image_feats = model.vit(pixel_values).last_hidden_state[:, 0, :]
                proj_feats = model.projector(image_feats).unsqueeze(1).to(torch.float16)

                prompts = [f"<image> Question: {q.strip()} Answer:" for q in questions]
                tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                input_embeds = model.phi.get_input_embeddings()(tokenized.input_ids)
                input_embeds = torch.cat([proj_feats, input_embeds], dim=1)

                attention_mask = torch.cat([
                    torch.ones((input_embeds.size(0), 1), dtype=torch.long).to(device),
                    tokenized.attention_mask
                ], dim=1)

                outputs = model.phi.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=50
                )

                decoded = tokenizer.batch_decode(outputs[:, 1:], skip_special_tokens=True)
                preds.extend(decoded)
                gts.extend(answers)

    anls_score = compute_anls(preds, gts)
    print(f"💡 Validation ANLS: {anls_score:.4f}")
    return anls_score

# ============ Main ============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("./phi-2")
    tokenizer.pad_token = tokenizer.eos_token

    vit = ViTModel.from_pretrained("./vit").eval().to(device)
    for p in vit.parameters(): p.requires_grad = False

    phi = AutoModelForCausalLM.from_pretrained("./phi-2", torch_dtype=torch.float16).eval().to(device)
    for p in phi.parameters(): p.requires_grad = False

    projector = nn.Linear(vit.config.hidden_size, phi.config.hidden_size).to(device)
    model = VisionLanguageModel(vit, phi, projector, tokenizer).to(device)

    train_dataset = FilteredDataset(DocVQADataset("train.csv", "train_images"))
    val_dataset = FilteredDataset(DocVQADataset("val.csv", "val_images"))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    train_model(model, train_loader, val_loader, tokenizer, device)
