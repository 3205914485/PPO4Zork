# 🧠 Zork LLM Agent with PPO

This repository contains code for training a language model agent to play the classic text-based game **Zork** using **Reinforcement Learning** with **Proximal Policy Optimization (PPO)**.

## 🏗️ Project Structure

```
.
├── sft/          # Supervised Fine-Tuning (SFT) code
├── ppo/          # PPO training code
├── evaluate/     # Evaluation and analysis scripts
├── data/         # Dataset files (walkthroughs, SFT data, reward annotations)
└── README.md
```

## 📘 Components

### 🔧 `sft/`
Scripts and configs for supervised fine-tuning the base language model to align with game-style action prediction.  
The model is trained to output the correct next action given a game state.

### 🤖 `ppo/`
Implements PPO-based reinforcement learning to improve the agent’s performance through environment interaction.  
Includes:
- GAE-based advantage estimation
- Actor-critic training
- RM or rule-based reward signal support
- PPO-specific tricks (reward norm, advantage norm, gradient clipping, etc.)

### 📊 `evaluate/`
Scripts to test trained models by playing episodes in the Zork environment and logging performance (e.g., score, room traversal, underground entry success, etc.).

## 💡 Highlights

- Supports training with either **Rule-Based** or **Reward Model (RM)** rewards
- Compatible with **Qwen** models (via HuggingFace)
- LoRA-compatible training for efficient fine-tuning
- Includes utilities for logging, evaluation, and visualizing reward curves

## 🚀 Getting Started

Install dependencies (e.g., `transformers`, `accelerate`, `peft`, `gym`, `zorklite`), prepare your model and dataset, then run SFT or PPO training.

## 📩 Contact

For questions, please reach out or open an issue.