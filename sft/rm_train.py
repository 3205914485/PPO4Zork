import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="qwen-2.5-1.5B-instruct",
                        help="预训练模型的路径")
    parser.add_argument("--dataset_path", type=str, default="sft/data/raw/zork1_rm_lora_dataset_penalty_argumented.jsonl",
                        help="处理后的数据集路径")
    parser.add_argument("--output_dir", type=str, default="sft/sft_model/qwen-2.5-1.5B_zork",
                        help="模型保存的输出路径")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批量大小")
    parser.add_argument("--max_length", type=int, default=1024, help="max_length of prompt")
    parser.add_argument("--epochs", type=int, default=3, help="训练的轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--device", type=str, default="cuda:1", help="设备")

    return parser.parse_args()


class RewardDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["context"]
        score = float(item["target"])
        encoded = self.tokenizer(prompt, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float)
        }


class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        last_token_idx = attention_mask.sum(1) - 1  # [batch]
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]  # [batch, hidden]
        rewards = self.value_head(last_hidden).squeeze(-1)  # [batch]
        return rewards


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)

            # 保留一位小数
            preds_rounded = torch.round(preds * 10) / 10
            scores_rounded = torch.round(scores * 10) / 10

            abs_errors = torch.abs(preds_rounded - scores_rounded)
            total_loss += abs_errors.sum().item()

        avg_loss = total_loss / len(dataloader)

    print(f"\n[Eval] Avg MAE Loss: {avg_loss:.4f}")
    
    model.train()


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    full_dataset = RewardDataset(args.dataset_path, tokenizer, args.max_length)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = RewardModel(args.model_name).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            score = batch["score"].to(args.device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(preds, score)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * args.gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} - Train Avg Loss: {avg_loss:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        evaluate(model, val_loader, args.device)

if __name__ == "__main__":
    args = parse_args()
    train(args)