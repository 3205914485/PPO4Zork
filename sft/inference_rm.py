import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from rm_train import RewardModel


def load_model_and_tokenizer(model_name, checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = RewardModel(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
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
        for step, line in tqdm(enumerate(infile)):
            data = json.loads(line)
            prompt = data.get("context", "")
            if prompt:
                score = inference(model, tokenizer, prompt, device, max_length=max_length)
                result = {
                    "step": step,
                    "score": score,
                    "context": prompt
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Reward Model from .pt checkpoint")
    parser.add_argument("--model_name", type=str, default='qwen-2.5-0.5B-instruct', help="e.g., qwen-2.5-1.5B-instruct")
    parser.add_argument("--checkpoint", type=str, default='sft/sft_model/qwen-2.5-0.5B-instruct_zork/checkpoint-epoch4.pt', help="Path to .pt checkpoint")
    parser.add_argument("--input_file", type=str, default='sft/data/raw/zork1_rm_lora_dataset_penalty_inference.jsonl', help="Input JSONL with `context` field")
    parser.add_argument("--output_file", type=str, default="sft/outputs/rm_output.jsonl", help="Where to save inference results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint, args.device)
    process_file(args.input_file, args.output_file, model, tokenizer, args.device, args.max_length)
    print(f"Finished inference. Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
