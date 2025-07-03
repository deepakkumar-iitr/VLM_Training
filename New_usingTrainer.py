import torch
import torch.nn as nn
import numpy as np
import editdistance
from transformers import ViTModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, TrainerCallback
from safetensors.torch import load_file, save_file

# --- ANLS Metric ---
def normalized_levenshtein(pred, gt):
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    dist = editdistance.eval(pred, gt)
    norm = dist / max(len(pred), len(gt))
    return 1 - norm

def compute_anls(preds, gts, threshold=0.5):
    scores = []
    for p, gt_list in zip(preds, gts):
        sims = [normalized_levenshtein(p, gt) for gt in gt_list]
        best_sim = max(sims)
        scores.append(best_sim if best_sim >= threshold else 0)
    return np.mean(scores)

# --- Custom Model ---
class VisionLanguageModel(nn.Module):
    def __init__(self, vit_path="./vit", phi_path="./phi-2"):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(vit_path, config=AutoConfig.from_pretrained(vit_path))
        self.vit.load_state_dict(load_file(f"{vit_path}/model.safetensors"))
        for p in self.vit.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(phi_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.phi = AutoModelForCausalLM.from_pretrained(phi_path, torch_dtype=torch.float16)
        self.phi.load_state_dict(load_file(f"{phi_path}/model.safetensors"), strict=False)
        for p in self.phi.parameters():
            p.requires_grad = False

        vit_dim = self.vit.config.hidden_size
        phi_dim = self.phi.config.hidden_size
        self.projector = nn.Linear(vit_dim, phi_dim)

    def forward(self, pixel_values, questions, answers=None):
        with torch.no_grad():
            vit_out = self.vit(pixel_values=pixel_values)
            cls_token = vit_out.last_hidden_state[:, 0, :]
        projected = self.projector(cls_token).unsqueeze(1).to(torch.float16)

        prompts = [f"<image> Question: {q} Answer:" for q in questions]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(projected.device)
        token_embeds = self.phi.get_input_embeddings()(inputs.input_ids).to(torch.float16)
        inputs_embeds = torch.cat([projected, token_embeds], dim=1)

        attention_mask = torch.cat([
            torch.ones((token_embeds.size(0), 1), dtype=torch.long).to(projected.device),
            inputs.attention_mask
        ], dim=1)

        if answers is not None:
            with self.tokenizer.as_target_tokenizer():
                label_ids = self.tokenizer(answers, return_tensors="pt", padding=True, truncation=True).input_ids.to(projected.device)
            labels = torch.full((label_ids.shape[0], inputs.input_ids.shape[1] + 1), -100).to(projected.device)
            labels[:, -label_ids.shape[1]:] = label_ids

            outputs = self.phi(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs.loss
        else:
            outputs = self.phi.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=50
            )
            return outputs

# --- Metric Function ---
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = model.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
    refs = [[gt.strip() for gt in label.split("|||")] for label in decoded_labels]
    score = compute_anls(decoded_preds, refs)
    return {"anls": score}

# --- Save Best Checkpoint Callback ---
class SaveBestProjectionCallback(TrainerCallback):
    def __init__(self, monitor="anls", save_path="best_projector.safetensors"):
        self.monitor = monitor
        self.save_path = save_path
        self.best_score = -1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        if score > self.best_score:
            self.best_score = score
            print(f"✅ New best {self.monitor}: {score:.4f}. Saving projector...")
            save_file(kwargs["model"].projector.state_dict(), self.save_path)

# --- Example Dataset + Trainer (not shown: dataset definition, args, etc.) ---
# model = VisionLanguageModel()
# args = TrainingArguments(...)
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=collate_fn,
#     tokenizer=model.tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[SaveBestProjectionCallback()]
# )
# trainer.train()
