import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
from Env import ZorkEnv 
from ppo_utils import compute_advantage
from MemoryAgent import MemoryAgent
from ReasoningAgent import ReasoningAgent
from PlanningAgent import PlanningAgent
from ActionAgent import ActionAgent

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
            "reasoning_prompt": prompts[i]['reasoning_prompt'],
            "planning_prompt": prompts[i]['planning_prompt'],
            "action_prompt": prompts[i]['action_prompt'],
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

    env = ZorkEnv("zork1.z5", max_repeat=30, max_history=args.max_history)

    # Load tokenizer & models
    tokenizer = AutoTokenizer.from_pretrained(args.am_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor, critic, reward_model = load_models(args)
    reasoning_agent = ReasoningAgent(actor, tokenizer, device=actor.device, max_tokens=256)
    planning_agent = PlanningAgent(actor, tokenizer, device=actor.device, max_tokens=256)
    action_agent = ActionAgent(actor, tokenizer, device=actor.device, max_tokens=16)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    scores_list = [0]

    for episode in range(args.max_episodes):
        print("Sampling trajectory")
        step_info = env.reset()
        memory_agent = MemoryAgent(current_location=step_info['location'])
        step = 0
        transitions = {"prompts": [], "locations": [], "inventories": [], "actions": [], "generates": [], "rewards": [], "logprobs": [], "values": []}
        step_bar = tqdm(total=args.max_epochs, desc=f"Episode {episode + 1}", ncols=150)
        end = None
        while not env.terminal and step < args.max_epochs:
            torch.cuda.empty_cache()
            memory = memory_agent.summarize()
            print(f"Memory: \n{memory}\n")
            reasoning_prompt = reasoning_agent.build_prompt(step_info['observation'], memory)
            thought, reason_logprobs = reasoning_agent.generate_thought(reasoning_prompt)
            print(f"Thought: \n{thought}\n")
            planning_prompt = planning_agent.build_prompt(step_info['observation'], memory, thought)
            goal, planning_logprobs, generated_texts_goal = planning_agent.plan(planning_prompt)
            print(f"Goal: \n{goal}\n")
            action_prompt = action_agent.build_prompt(thought, goal, candidates=step_info['candidates'])        
            action, action_logprobs = action_agent.decide_action(action_prompt)            

            step_info['action'] = action
            location = step_info['location']
            inventory = step_info['inventory']
            critic_prompt = build_critic_prompt(env.history, step_info)
            value = estimate_value(tokenizer, critic, critic_prompt)
            step_info = env.step(action)

            memory_agent.update(action, step_info['location'], step_info['inventory'])

            reward = step_info['reward']

            transitions["prompts"].append({
                            "reasoning_prompt":reasoning_prompt,
                            "planning_prompt":planning_prompt,
                            "action_prompt":action_prompt,
                            "critic_prompt":critic_prompt
                            })
            transitions["generates"].append({
                            "thought":thought,
                            "goal":generated_texts_goal
                            })            

            transitions["locations"].append(location)
            transitions["inventories"].append(inventory)
            transitions["actions"].append(action)
            transitions["logprobs"].append([reason_logprobs, planning_logprobs, action_logprobs])
            transitions["rewards"].append(reward)
            transitions["values"].append(value)
            end = step_info['observation']
            step += 1
            step_bar.update(1)
            step_bar.set_postfix({"reward": reward, "location": location, "action": action[:30]})

            if env.check_stuck(action, location):
                print("The agent has been stuck !!")
                break


        step_bar.close()
        score = env.score
        total_reward = sum(transitions["rewards"])
        print(f"End: {end}")
        print(f"Score: {score}")
        print("Sampling Done")

        ppo_epochs = args.ppo_epochs
        if score > max(scores_list):
            ppo_epochs *= 3
        scores_list.append(score)

        # GAE advantage & returns
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float)
        values = torch.tensor(transitions["values"], dtype=torch.float)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # GAE
        td_target = rewards + args.gamma * torch.cat([values[1:], torch.tensor([0.0])])
        td_delta = td_target - values
        advantages = compute_advantage(args.gamma, args.lmbda, td_delta)

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # critic target
        returns = advantages + values
        
        transitions['values'] = list(values.numpy())
        transitions['advantages'] = list(advantages.numpy())
        transitions['returns'] = list(returns.numpy())

        save_trajectory(episode + 1, transitions, save_dir=args.trj_save_path)

        # PPO train
        torch.cuda.empty_cache()
        thought_losses = []
        goal_losses = []
        action_losses = []
        critic_losses = []
        actor.train()
        critic.train()
        print_trainable_parameters(actor, "Actor")
        print_trainable_parameters(critic, "Critic")
        if args.pretrain_critic and episode <= args.pretrain_critic_episodes:
            print('Still Pretraining the critic')

        for _ in tqdm(range(ppo_epochs)):
            for j in tqdm(range(len(transitions["actions"]))):

                prompt = transitions["prompts"][j]
                generate = transitions["generates"][j]
                location = transitions["locations"][j]
                inventory = transitions["inventories"][j]
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
                    thought_loss = reasoning_agent.update(prompt['reasoning_prompt'], generate['thought'], old_log_prob[0], advantage, actor_optimizer, args.clip_eps)
                    thought_losses.append(thought_loss)
                    goal_loss = planning_agent.update(prompt['planning_prompt'], generate['goal'], old_log_prob[1], advantage, actor_optimizer, args.clip_eps)
                    goal_losses.append(goal_loss)
                    action_loss = action_agent.update(prompt['action_prompt'], action, old_log_prob[2], advantage, actor_optimizer, args.clip_eps)
                    action_losses.append(action_loss)

        # === Episode logging ===
        avg_thought_loss = sum(thought_losses) / len(thought_losses) if thought_losses else 0
        avg_goal_loss = sum(goal_losses) / len(goal_losses) if goal_losses else 0
        avg_action_loss = sum(action_losses) / len(action_losses) if action_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        avg_advantage = advantages.mean().item()

        print(f"[Ep {episode + 1}] Reward: {total_reward:.2f} | Score: {env.score} | Action Loss: {avg_action_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

        with open(log_file, "a") as f:
            f.write(f"{episode+1:7d} | {total_reward:13.4f} | {env.score:9d} | {avg_action_loss:11.6f} | {avg_critic_loss:12.6f} | {avg_advantage:15.6f}\n")

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
    parser.add_argument("--critic_lr", type=float, default=1e-5)
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
    args.prefix_path = os.path.join("mutil_agents/saving", args.prefix)
    args.save_path = os.path.join(args.prefix_path, "ckpts")
    args.log_save_path = os.path.join(args.prefix_path, "logs")
    args.trj_save_path = os.path.join(args.prefix_path, "trajectories")

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_save_path, exist_ok=True)
    os.makedirs(args.trj_save_path, exist_ok=True)

    os.makedirs(args.save_path, exist_ok=True)
    ppo_loop(args)