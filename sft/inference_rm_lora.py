import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch.nn as nn
import numpy as np
import os

class RewardModelWithLoRA(nn.Module):
    def __init__(self, base_model_name_or_path, lora_path):
        super().__init__()
        # 加载 tokenizer-compatible base 模型
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        # 注入 LoRA 权重
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        # 加载 value head 参数
        value_head_path = os.path.join(lora_path, "value_head.pt")
        self.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))

    def forward(self, input_ids, attention_mask):
        # 注意这里：output_hidden_states=True 以便我们拿 hidden state
        outputs = self.model.base_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # 最后一层 hidden state
        last_token_idx = attention_mask.sum(1) - 1
        final_hidden = last_hidden[torch.arange(last_hidden.size(0)), last_token_idx]
        reward = self.value_head(final_hidden).squeeze(-1)
        return reward


def load_model_and_tokenizer(base_model_name, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = RewardModelWithLoRA(base_model_name, lora_path).to(device)
    model.eval()
    return model, tokenizer


def inference(model, tokenizer, prompt, device, max_length=512):
    inputs = tokenizer(prompt, padding="max_length", truncation=True,
                       max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        score = model(input_ids=input_ids, attention_mask=attention_mask)
    return score.item()


def process_file(input_file, output_file, model, tokenizer, device, max_length):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        acc = []
        for step, line in tqdm(enumerate(infile)):
            data = json.loads(line)
            prompt = data.get("context", "")
            target = data.get("target","")
            if prompt:
                score = inference(model, tokenizer, prompt, device, max_length=max_length)
                score = round(score, 1)
                result = {
                    "step": step,
                    "score": score,
                    "context": prompt
                }
                acc.append(int(round(float(target), 1) == score))
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.write(f"Total Acc: {np.mean(acc)}" + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Reward Model with LoRA adapter")
    parser.add_argument("--model_name", type=str, default='qwen-2.5-1.5B-instruct', help="Path to base model")
    parser.add_argument("--lora_path", type=str, default='sft/sft_model/qwen-2.5-1.5B-instruct_zork_lora/lora_epoch30', help="Path to LoRA adapter folder")
    parser.add_argument("--input_file", type=str, default='sft/data/raw/zork1_rm_lora_dataset_penalty.jsonl', help="Input JSONL with `context` field")
    parser.add_argument("--output_file", type=str, default="sft/outputs/rm_output.jsonl", help="Where to save inference results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.lora_path, args.device)
    process_file(args.input_file, args.output_file, model, tokenizer, args.device, args.max_length)
    print(f"Finished inference. Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
