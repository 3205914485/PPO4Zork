import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model_and_tokenizer(model_name, device, use_8bit, mode):
    model_path = f"{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map=device, load_in_8bit=use_8bit
    )

    if mode == 'lora':
        print("LoRA mode enabled.")
        peft_model_path = f"sft/sft_model/{model_name}_zork"
        model = PeftModel.from_pretrained(model, peft_model_path)

    model.eval()
    return model, tokenizer


def inference(model, tokenizer, prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_file(input_file, output_file, model, tokenizer, max_length):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for step, line in tqdm(enumerate(infile)):
            data = json.loads(line) 
            prompt = data.get("context", "")
            if prompt:
                # print(prompt)
                response = inference(model, tokenizer, prompt, max_length=max_length)
                outfile.write(f"STEP: {step}\n{response}\n\n\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for LLM-Zork model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g., qwen-2.5-3B-instruct")
    parser.add_argument("--input_file", type=str, default="sft/data/raw/zork1_wt_inference.jsonl", help="Input JSONL file with `context` field")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum generation length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model")
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "base"], help="Inference mode")
    parser.add_argument("--use_8bit", type=bool, default=False, help="Use 8-bit quantization")
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.use_8bit, args.mode)

    # 自动构造输出路径
    output_suffix = "_org.txt" if args.mode == "base" else ".txt"
    output_file = f"sft/outputs/{args.model_name}{output_suffix}"

    process_file(args.input_file, output_file, model, tokenizer, args.max_length)
    print(f"Finished inference. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
