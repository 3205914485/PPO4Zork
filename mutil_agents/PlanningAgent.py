import torch
import re

class PlanningAgent:
    def __init__(self, model, tokenizer, device='cuda:0', max_tokens=128):
        self.model = model  # ActorModel or similar
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.goals = []

    def build_prompt(self, observation: str, memory_summary: str, thought: str) -> str:
        return (
            "You are an intelligent planner in the text-based game Zork.\n"
            "The game target is to explore the world, collect treasures, solve puzzles, and maximize your score.\n"
            "You need to plan what to do step by step"
            "Given the current situation, memory, reasoning and your previous goals, decide your current goal.\n\n"
            f"--- Observation ---\n{observation.strip()}\n\n"
            f"--- Memory ---\n{memory_summary.strip()}\n\n"
            f"--- Reasoning ---\n{thought.strip()}\n\n"
            f"--- Previous Goal ---\n{self.goals or 'None'}\n\n"
            "Now write your planning analysis, then end with:\n<goal>your short goal here</goal>"
        )

    def plan(self, prompt, max_retries=5) -> tuple[str, float, str]:
        """
        Returns:
            goal (str): content inside <goal>...</goal>
            log_prob (float): log-prob of full generated sequence
            full_text (str): full output including analysis
        """
        for attempt in range(max_retries):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[-1]

            with torch.no_grad():
                outputs = self.model.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=10,
                    top_p=0.9,
                    temperature=0.8,
                )
                generated = outputs[0][input_len:]
                generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)

            # 尝试提取 <goal>...</goal>
            match = re.search(r"<goal>(.*?)</goal>", generated_text, re.DOTALL)
            if match:
                goal = match.group(1).strip()
                full_text = prompt + generated_text

                full_input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    logits = self.model.model(input_ids=full_input_ids).logits[:, :-1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                generated_ids = self.tokenizer(generated_text, return_tensors="pt").input_ids[:, 1:].to(self.device)
                selected_logprobs = log_probs[:, -generated_ids.size(1):].gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
                log_prob = selected_logprobs.sum(dim=-1).item()

                self.goals.append(goal)
                return goal, log_prob, generated_text 

            print(f"[PlanningAgent] No <goal> match, retrying ({attempt+1}/{max_retries})...")

        print("[PlanningAgent] Failed to extract goal. Using fallback.")
        self.goals.append(generated_text.strip())
        return generated_text.strip(), 0.0, generated_text.strip()  # fallback


    def update(self, prompt, goal, old_logprobs, advantage, opt, clip_eps):
        inputs = self.tokenizer(prompt + goal, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # new policy log_prob
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :].cpu()  # remove last token (predict next) [bs, seqlen, vocab]
        probs = torch.log_softmax(logits, dim=-1) 

        generate_ids = self.tokenizer(goal, return_tensors="pt").input_ids[:, 1:]
        selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
        log_prob = selected_logprobs.sum(dim=-1)

        # PPO clipped loss
        ratio = torch.exp(log_prob - old_logprobs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

        goal_loss = -torch.min(surr1, surr2).mean()
        if torch.isnan(goal_loss) or torch.isinf(goal_loss):
            print("Found NaN in goal_loss, skipping step")
            return 0
        opt.zero_grad()
        goal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        opt.step()
        return goal_loss
