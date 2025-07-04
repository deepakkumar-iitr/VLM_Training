import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer,
    AutoModelForCausalLM, ViTModel
)
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from typing import List
from datasets import load_metric

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
        image = Image.open(os.path.join(self.images_dir, row["imageid"])).convert("RGB")
        return {
            "image": self.transform(image),
            "question": row["question"],
            "answers": row["answers"].split("|||")
        }

# ============ Collate Function ============
def collate_fn(batch):
    pixel_values = torch.stack([item["image"] for item in batch])
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]  # List[List[str]]
    return {
        "pixel_values": pixel_values,
        "questions": questions,
        "answers": answers
    }

# ============ ANLS Metric ============
import editdistance

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
        )

# ============ Compute Metrics ============
def compute_metrics(eval_preds):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    tokenizer = compute_metrics.tokenizer

    preds = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {"anls": compute_anls(decoded_preds, [[label] for label in decoded_labels])}

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

    model = VisionLanguageModel(vit, phi, projector, tokenizer)
    compute_metrics.tokenizer = tokenizer  # inject tokenizer for decoding

    train_dataset = DocVQADataset("train.csv", "train_images")
    val_dataset = DocVQADataset("val.csv", "val_images")

    training_args = TrainingArguments(
        output_dir="./vlm_output",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="anls",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Save projector
    from safetensors.torch import save_file
    save_file(projector.state_dict(), "./vlm_output/projector_best.safetensors")


