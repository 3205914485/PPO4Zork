import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from Env import ZorkEnv 
from ppo_utils import compute_advantage  
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import os
import json
from tqdm import tqdm
from collections import Counter
import math
import re

from models import load_models

SYSTEM_ACTOR_PROMPT = (
    "You are playing the interactive fiction game Zork.\n"
    "Your goal is to explore the world, collect treasures, solve puzzles, and maximize your score.\n"
    "You interact with the world by typing commands like \"go north\", \"open door\", \"take lamp\", etc.\n"
    "You see the world through textual descriptions (observations), and your inventory tells you what you're carrying.\n"
    "At each step, you will be shown some candidate actions for reference.\n"
    "You can either choose one of them, or generate your own appropriate action to proceed.\n\n"
)

SYSTEM_CRITIC_PROMPT = (
    "You are a value estimation model in a text-based interactive fiction game.\n"
    "Your task is to evaluate the quality of a decision-making context, based on the current locations and past actions.\n"
    "You will be given a history of previous locations and actions, along with the current state.\n"
    "Your job is to assess how promising this context is in terms of achieving high rewards in the game.\n\n"
)

SYSTEM_REWARD_PROMPT = (
    "You are a reward model trained to evaluate the quality of actions taken in the game Zork.\n"
    "You are given the player's current location, their inventory, and the action they just performed.\n"
    "Your task is to score how beneficial the action is on a scale from 0 to 1, where:\n"
    "- 0 means the action is useless or redundant\n"
    "- 1 means the action is extremely valuable for game progression\n\n"
    "You should reward actions that:\n"
    "- Collect or use key items (e.g., lamp, keys)\n"
    "- Explore new areas\n"
    "- Solve puzzles (e.g., moving rug, praying at mirror)\n"
    "- Defeat enemies\n"
    "- Store treasures\n"
    "- Progress the story\n\n"
    "You should penalty actions that:\n"
    "- Useless direction move and repeat\n"
)

def print_trainable_parameters(model, model_name="Model"):
    total_params = 0
    trainable_params = 0
    lora_params = 0

    print(f"=== {model_name} Trainable Parameters Summary ===")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            if 'lora' in name.lower():
                lora_params += num_params

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,} ({trainable_params / total_params:.4%})")
    print(f"LoRA Parameters      : {lora_params:,} ({lora_params / trainable_params:.4%} of trainable)")
    print("=" * 60)

def save_trajectory(episode_idx, transistions, save_dir="ppo_ckpts"):
    prompts = transistions['prompts']
    actions = transistions['actions']
    rewards = transistions['rewards']
    values = transistions['values']
    advantages = transistions['advantages']
    returns = transistions['returns']
    trajectory = []
    for i in range(len(prompts)):
        item = {
            "step": i,
            "prompts": prompts[i]['actor_prompt'],
            "action": actions[i],
            "reward": rewards[i],
            "values": float(values[i]),
            "advantages": float(advantages[i]),
            "returns": float(returns[i])
        }   
        trajectory.append(item)
    
    path = os.path.join(save_dir, f"trajectory_ep{episode_idx}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)

def build_actor_prompt(history, current):
    prompt = SYSTEM_ACTOR_PROMPT
    if history:
        prompt += "--- Previous Actions & Observations ---\n"
        for line in history:
            prompt += f"{line['observation'].strip()} \n>{line['action'].strip()}\n"
        prompt += "\n"
    prompt += "--- Current Observation ---\n"
    prompt += f"Observation: {current['observation'].strip()}\n"
    prompt += f"Inventory: {current['inventory']}\n"

    if current['candidates']:
        prompt += "Choices:\n"
        for i, choice in enumerate(current['candidates'], 1):
            prompt += f"({i}) {choice}\n"

    prompt += "What should you do next?\nFirst, think carefully and reason about the situation within a <think>...</think> block.\nThen, provide your final action within an <answer>...</answer> block."
    return prompt

def build_critic_prompt(history, current):
    # prompt = SYSTEM_CRITIC_PROMPT
    prompt = ''
    # if history:
    #     prompt += "--- Previous Actions & Observations ---\n"
    #     for line in history:
    #         prompt += f"location:{line['location'].strip()} \n inventory:{line['inventory']} \n>action:{line['action'].strip()}\n\n"
    #     prompt += "\n"

    prompt += "--- Current Observation ---\n"
    prompt += f"Location: {current['location']}\n"
    prompt += f"Inventory: {current['inventory']}\n"
    # prompt += f"Action: {current['action']}\n\n"
    # prompt += "Score this action from 0 to 1:\nAnswer:"
    return prompt

def build_reward_prompt(history, current):
    prompt = SYSTEM_REWARD_PROMPT
    if history:
        prompt += "--- Previous Actions & Observations ---\n"
        for line in history:
            prompt += f"location:{line['location'].strip()} \n inventory:{line['inventory']} \n>action:{line['action'].strip()}\n\n"
        prompt += "\n"

    prompt += "--- Current Observation ---\n"
    prompt += f"Location: {current['location']}\n"
    prompt += f"Inventory: {current['inventory']}\n"
    prompt += f"Action: {current['action']}\n\n"
    prompt += "Score this action from 0 to 1:\nAnswer:"
    return prompt

def generate_action(tokenizer, actor, prompt, max_length=2048):
    with torch.no_grad():
        # 1. sample output
        inputs = tokenizer(prompt, return_tensors="pt").to(actor.device)
        input_len = inputs.input_ids.shape[-1]

        if input_len > max_length - 384:
            inputs.input_ids = inputs.input_ids[:, -(max_length - 384):]
            inputs.attention_mask = inputs.attention_mask[:, -(max_length - 384):]
            input_len = inputs.input_ids.shape[-1]

        outputs = actor.model.generate(
            **inputs,
            max_new_tokens=384,  # 够用
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.8
        )
        generated = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print('\n'+generated_text+'\n')

        # 2. extract <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if match:
            action = match.group(1).strip()
            success_flag = 1
        else:
            action = generated_text[:3]  # fallback，取前3个字符
            success_flag = 0

        # 3. calculate log prob for the **full generated_text**
        full_input = prompt + generated_text
        full_input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(actor.device)
        outputs = actor.model(input_ids=full_input_ids)
        logits = outputs.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        generated_ids = tokenizer(generated_text, return_tensors="pt").input_ids[:, 1:].to(actor.device)
        selected_logprobs = log_probs[:, -generated_ids.size(1):].gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
        action_log_prob = selected_logprobs.sum(dim=-1).item()

        return generated_text, action, action_log_prob, success_flag


def estimate_reward(tokenizer, reward_model, prompt, max_length=1024):
    with torch.no_grad():
        inputs = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(reward_model.device)
        attention_mask = inputs["attention_mask"].to(reward_model.device)
        score = reward_model(input_ids=input_ids, attention_mask=attention_mask)
    return score.item()

def estimate_value(tokenizer, critic_model, prompt, max_length=1024):
    with torch.no_grad():
        inputs = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(critic_model.device)
        attention_mask = inputs["attention_mask"].to(critic_model.device)
        value = critic_model(input_ids=input_ids, attention_mask=attention_mask)
    return value.item()


def ppo_loop(args):

    # log
    log_file = os.path.join(args.log_save_path, f"log.txt")
    with open(log_file, "w") as f:
        f.write(f"{'Episode':>7} | {'Total Reward':>13} | {'Env Score':>9} | {'Actor Loss':>11} | {'Critic Loss':>12} | {'Advantage Mean':>15}\n")
        f.write("-" * 78 + "\n")

    # Load tokenizer & models
    tokenizer = AutoTokenizer.from_pretrained(args.am_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor, critic, reward_model = load_models(args)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    scores_list = [0]
    env = ZorkEnv("zork1.z5", max_repeat=30, max_history=args.max_history)
    # Sample trajetories
    best_trajectory = None
    best_total_reward = 0
    best_score = 0
    for episode in range(args.max_episodes):
        print("Sampling trajectory")
        step_info = env.reset()
        step = 0
        transitions = {"prompts": [], "actions": [], "generates": [], "rewards": [], "logprobs": [], "values": [], "thinking": []}
        step_bar = tqdm(total=args.max_epochs, desc=f"Episode {episode + 1}", ncols=150)
        thinking_flags = []
        end = None
        while not env.terminal and step < args.max_epochs:

            actor_prompt = build_actor_prompt(env.history, step_info)
            generate, action, log_prob, thinking_flag = generate_action(tokenizer, actor, actor_prompt)
            step_info['action'] = action
            location = step_info['location']
            thinking_flags.append(thinking_flag)

            if thinking_flag:
                critic_prompt = build_critic_prompt(env.history, step_info)
                value = estimate_value(tokenizer, critic, critic_prompt)
                step_info = env.step(action)
                reward = step_info['reward']
                transitions["prompts"].append({
                                "actor_prompt":actor_prompt,
                                "critic_prompt":critic_prompt
                                })
                transitions["actions"].append(action)
                transitions["logprobs"].append(log_prob)
                transitions["rewards"].append(reward)
                transitions["values"].append(value)
                end = step_info['observation']
                step += 1
                step_bar.update(1)
                step_bar.set_postfix({"reward": reward, "location": location, "action": action[:30]})

            transitions["generates"].append(generate)                            
            transitions["thinking"].append(thinking_flag)

            if not thinking_flag:
                print("Do Not thinking successfully! ")

            if env.check_stuck(action, location):
                print("The agent has been stuck !!")
                break


        step_bar.close()
        score = env.score
        total_reward = sum(transitions["rewards"])
        print(f"End: {end}")
        print(f"Score: {score}")
        print("Sampling Done")
        if not args.pretrain_critic or episode > args.pretrain_critic_episodes:
            if score >= best_score:
                print("Best trajetory updated")
                best_total_reward = total_reward
                best_score = score
                best_trajectory = {
                    "prompts": transitions["prompts"],
                    "actions": transitions["actions"],
                    "generates": transitions["generates"]
                }

        ppo_epochs = args.ppo_epochs
        if score > max(scores_list):
            ppo_epochs *= 3
        scores_list.append(score)

        # GAE advantage & returns
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float)
        values = torch.tensor(transitions["values"], dtype=torch.float)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # rewards = torch.clamp(rewards, -1, 1)  

        # GAE
        td_target = rewards + args.gamma * torch.cat([values[1:], torch.tensor([0.0])])
        td_delta = td_target - values
        advantages = compute_advantage(args.gamma, args.lmbda, td_delta)

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # advantages = (advantages - advantages.mean())
        # advantages = torch.clamp(advantages, -1, 1)
        # critic target
        returns = advantages + values
        
        transitions['values'] = list(values.numpy())
        transitions['advantages'] = list(advantages.numpy())
        transitions['returns'] = list(returns.numpy())

        save_trajectory(episode + 1, transitions, save_dir=args.trj_save_path)
        # PPO train
        actor_losses = []
        critic_losses = []
        actor.train()
        critic.train()
        print_trainable_parameters(actor, "Actor")
        print_trainable_parameters(critic, "Critic")
        if args.pretrain_critic and episode <= args.pretrain_critic_episodes:
            print('Still Pretraining the critic')

        for _ in tqdm(range(ppo_epochs)):
            j = 0
            for i in tqdm(range(len(transitions["thinking"]))):
                thinking_flag = transitions["thinking"][i]
                generate = transitions['generates'][i]
                
                if thinking_flag:
                    # --- critic ---
                    prompt = transitions["prompts"][j]
                    action = transitions["actions"][j]
                    old_log_prob = transitions["logprobs"][j]
                    advantage = advantages[j]
                    target = returns[j]
                    critic_inputs = tokenizer(prompt['critic_prompt'], return_tensors="pt").to(critic.device)
                    value_pred = critic(**critic_inputs) ## forward -> scaler
                    target = target.to(critic.device).to(value_pred.dtype)  
                    critic_loss = F.mse_loss(value_pred.squeeze(), target)
                    if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                        print("Found NaN in critic loss, skipping step")
                        continue
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                    critic_optimizer.step()
                    critic_losses.append(critic_loss.item())
                    
                    # ---  actor ---
                    if not args.pretrain_critic or episode > args.pretrain_critic_episodes:

                        inputs = tokenizer(prompt['actor_prompt'] + generate, return_tensors="pt", padding=True).to(actor.model.device)
                        input_ids = inputs.input_ids
                        attention_mask = inputs.attention_mask

                        # new policy log_prob
                        output = actor(input_ids=input_ids, attention_mask=attention_mask)
                        logits = output.logits[:, :-1, :].cpu()  # remove last token (predict next) [bs, seqlen, vocab]
                        probs = torch.log_softmax(logits, dim=-1) 

                        generate_ids = tokenizer(generate, return_tensors="pt").input_ids[:, 1:]
                        selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
                        log_prob = selected_logprobs.sum(dim=-1)

                        # PPO clipped loss
                        ratio = torch.exp(log_prob - old_log_prob)
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantage
                    
                        actor_loss = -torch.min(surr1, surr2).mean()
                        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                            print("Found NaN in actor loss, skipping step")
                            continue
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                        actor_optimizer.step()
                        actor_losses.append(actor_loss.item())

                    j += 1

                else:
                    if not args.pretrain_critic or episode > args.pretrain_critic_episodes:
                        # ---  actor ---
                        inputs = tokenizer(prompt['actor_prompt'] + generate, return_tensors="pt").to(actor.model.device)
                        input_ids = inputs.input_ids
                        attention_mask = inputs.attention_mask

                        # forward
                        output = actor(input_ids=input_ids, attention_mask=attention_mask)
                        logits = output.logits[:, :-1, :].cpu()
                        probs = torch.log_softmax(logits, dim=-1)

                        generate_ids = tokenizer(generate, return_tensors="pt").input_ids[:, 1:]

                        selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
                        log_prob = selected_logprobs.sum(dim=-1)

                        actor_loss = log_prob.mean()  
                        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                            print("Found NaN in actor loss, skipping step")
                            continue
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                        actor_optimizer.step()
                        actor_losses.append(actor_loss.item())
        
        # BC best trajectory
        if best_trajectory is not None:
            print(f"Behavior Cloning on Best Trajectory with Score: {best_score}")
            for prompt, generate in zip(best_trajectory["prompts"], best_trajectory["generates"]):
                inputs = tokenizer(prompt['actor_prompt'] + generate, return_tensors="pt", padding=True).to(actor.model.device)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                output = actor(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits[:, :-1, :]
                probs = torch.log_softmax(logits, dim=-1)
                generate_ids = tokenizer(generate, return_tensors="pt").input_ids[:, 1:].to(actor.model.device)
                selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
                bc_loss = -selected_logprobs.mean()
                if torch.isnan(bc_loss) or torch.isinf(bc_loss):
                    print("Found NaN in BC loss, skipping")
                    continue
                actor_optimizer.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_optimizer.step()

        # === Episode logging ===
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        avg_advantage = advantages.mean().item()

        print(f"[Ep {episode + 1}] Reward: {total_reward:.2f} | Score: {env.score} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

        with open(log_file, "a") as f:
            f.write(f"{episode+1:7d} | {total_reward:13.4f} | {env.score:9d} | {avg_actor_loss:11.6f} | {avg_critic_loss:12.6f} | {avg_advantage:15.6f}\n")

        # 保存模型
        if (episode + 1) % args.save_every == 0:
            actor.model.save_pretrained(os.path.join(args.save_path, f"actor_ep{episode+1}"))
            torch.save(critic.value_head, os.path.join(args.save_path, f"critic_ep{episode+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Config
    parser.add_argument("--am_path", type=str, default="/data3/whr/zst/huggingface/qwen-2.5-3B-instruct")
    parser.add_argument("--am_lora_path", type=str, default=None)
    parser.add_argument("--rm_lora", type=int, default=0)
    parser.add_argument("--rm_path", type=str, default="/data3/whr/zst/huggingface/qwen-2.5-1.5B-instruct")
    parser.add_argument("--rm_ckpts", type=str, default="sft/sft_model/qwen-2.5-1.5B-instruct_zork_lr1e-5/checkpoint-epoch29.pt")
    parser.add_argument("--rm_lora_path", type=str, default="sft/sft_model/qwen-2.5-1.5B-instruct_zork_lora_rk10/lora_epoch50")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cm_use_rm", type=int, default=1)
    parser.add_argument("--cm_use_table", type=int, default=0)
    parser.add_argument("--am_device", type=str, default='cuda:0')
    parser.add_argument("--cm_device", type=str, default='cuda:0')
    parser.add_argument("--rm_device", type=str, default='cuda:0')

    # Training Config
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=500, help='for a trajectory')
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--max_history", type=int, default=3)
    parser.add_argument("--critic_lr", type=float, default=5e-6)
    parser.add_argument("--actor_lr", type=float, default=3e-6)
    parser.add_argument("--rm_use", type=int, default=1)

    # PPO Config
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--pretrain_critic", type=int, default=0)
    parser.add_argument("--pretrain_critic_episodes", type=int, default=5)
    # Saving Config
    parser.add_argument("--prefix", type=str, default="test", help="prefix name for this run (creates ppo/{prefix}/...)")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    # args.am_device = args.device
    args.rm_device = args.device
    args.cm_device = args.device
    args.prefix_path = os.path.join("ppo/saving", args.prefix)
    args.save_path = os.path.join(args.prefix_path, "ckpts")
    args.log_save_path = os.path.join(args.prefix_path, "logs")
    args.trj_save_path = os.path.join(args.prefix_path, "trajectories")

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_save_path, exist_ok=True)
    os.makedirs(args.trj_save_path, exist_ok=True)

    os.makedirs(args.save_path, exist_ok=True)
    ppo_loop(args)