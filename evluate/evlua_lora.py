import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# import envrionment
from jericho import FrotzEnv

SYSTEM_PROMPT = (
    "You are playing the interactive fiction game Zork.\n"
    "Your goal is to explore the world, collect treasures, solve puzzles, and maximize your score.\n"
    "You interact with the world by typing commands like \"go north\", \"open door\", \"take lamp\", etc.\n"
    "You see the world through textual descriptions (observations), and your inventory tells you what you're carrying.\n"
    "At each step, you will be shown some candidate actions for reference.\n"
    "You can either choose one of them, or generate your own appropriate action to proceed.\n\n"
)

class zork1:
    def __init__(self, env_name, max_history=3, max_repeat=10):
        path = f"../z-machine-games-master/jericho-game-suite/{env_name}.z5"
        self.env = FrotzEnv(path)
        self.max_history = max_history
        self.max_repeat = max_repeat
        self.history = []
        self.terminal = False
        self.score = 0
        self.action_history = []

    def reset(self):
        obs, info = self.env.reset()
        self.terminal = False
        self.history = []
        self.score = 0
        reward = 0
        inventory = []
        valued_actions = self.env.get_valid_actions()
        step_info = {
            "observation": obs,
            "inventory": inventory,
            "candidates": valued_actions,
            "action": None
        }
        self.history.append(step_info)
        prompt = self.build_prompt({**step_info})
        return obs, reward, self.score, prompt

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.terminal = done
        self.score = info['score']
        inventory = [item.name for item in self.env.get_inventory()]
        valued_actions = self.env.get_valid_actions()
        step_info = {
            "observation": obs,
            "inventory": inventory,
            "candidates": valued_actions,
            "action": None
        }
        # input the a_{i-1} to last history
        self.history[-1]['action'] = action
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(step_info)
        prompt = self.build_prompt({"history": self.history[:-1], **step_info})
        return obs, reward, self.score, prompt

    def build_prompt(self, item):
        prompt = SYSTEM_PROMPT
        if item.get("history"):
            prompt += "--- Previous Actions & Observations ---\n"
            for step in item["history"]:
                prompt += f"{step['observation'].strip()} \n> {step['action'].strip()}\n"
            prompt += "\n"

        prompt += "--- Current Observation ---\n"
        prompt += f"Observation: {item['observation']}\n"
        prompt += f"Inventory: {item['inventory']}\n"

        if item["candidates"]:
            prompt += "Choices:\n"
            for i, choice in enumerate(item["candidates"], 1):
                prompt += f"({i}) {choice}\n"
        prompt += "\nWhat should you do next?\nAnswer:"
        return prompt

    def check_stuck(self, action):
        self.action_history.append(action)
        
        if len(self.action_history) > self.max_repeat:
            recent_actions = self.action_history[-self.max_repeat:]
            if len(set(recent_actions)) <= 3:
                return True

        return False


def load_model_and_tokenizer(model_name, device, mode):
    model_path = f"/data3/whr/zhk/huggingface/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map=device
    )

    if mode == 'lora':
        print("LoRA mode enabled.")
        peft_model_path = f"sft/sft_model/{model_name}_zork"
        model = PeftModel.from_pretrained(model, peft_model_path)

    model.eval()
    return model, tokenizer


def inference(model, tokenizer, prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length = max_length,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            top_k = 20,
            top_p = 0.7,
            temperature=1.1
        )
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()



def go(env, output_file, model, tokenizer, max_length, max_epochs):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        observation, reward, score, prompt = env.reset()
        outfile.write(f"{observation}\n\n\n")
        step = 0
        print("GAME START !!!")
        while not env.terminal and step < max_epochs:
            response = inference(model, tokenizer, prompt, max_length=max_length)
            observation, reward, score, prompt = env.step(response)
            outfile.write(f"STEP: {step}\n")
            outfile.write(f"Action: {response}\n")
            outfile.write(f"Observation:\n{observation}\n")
            outfile.write(f"Reward: {reward}\n")
            outfile.write(f"Score: {score}\n\n\n")
            step += 1
            print(f"Step Now: {step}")
            print(f"LLM: {response}\n")
            print(f"Observation:\n{observation}\n")
            print(f"Reward: {reward}\n\n")
            if env.check_stuck(response):
                print("The agent has been stuck !!")
                break
    return score

          
def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for LLM-Zork model")
    parser.add_argument("--model_name", type=str, default='qwen-2.5-7B-instruct', help="Model name, e.g., qwen-2.5-7B-instruct")
    parser.add_argument("--output_file", type=str, default='evluate/outputs/sft_lora/qwen-2.5-7B-instruct/000.txt', help="output file path")
    parser.add_argument("--env_name", type=str, default='zork1', help="Envrionment name, e.g., zork1")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum generation length")
    parser.add_argument("--max_history", type=int, default=3, help="Maximum prompt generation history")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum epochs num for one trajectory")
    parser.add_argument("--max_repeat", type=int, default=30, help="Maximum repeat action for checking stuck")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run the model")
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "base"], help="Inference mode")
    return parser.parse_args()


def main():
    
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.mode)

    env = zork1(env_name=args.env_name, max_history=args.max_history, max_repeat=args.max_repeat)

    score = go(env, args.output_file, model, tokenizer, args.max_length, args.max_epochs)
    print(f"Finished inference. Results saved to: {args.output_file}")
    print(score)


if __name__ == "__main__":
    main()
