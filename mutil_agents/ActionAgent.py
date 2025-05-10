import torch
import re

class ActionAgent:
    def __init__(self, model, tokenizer, device='cuda:0', max_tokens=12):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def build_prompt(self, thought: str, goal: str, candidates: list[str]) -> str:
        prompt = (
            "You are an intelligent agent playing the game Zork.\n"
            "You are given your reasoning, and your planning goal.\n"
            "You interact with the world by typing commands like \"go north\", \"open door\", \"take lamp\", etc.\n"
            "You are also provided with candidate actions that you may choose from or use for inspiration.\n"
            "Pick the most appropriate action, and just output the action you chosed without any other information!.\n\n"
            f"--- Reasoning ---\n{thought.strip()}\n\n"
            f"--- Goal ---\n{goal.strip()}\n\n"
        )

        if candidates:
            prompt += "--- Candidate Actions ---\n"
            for i, cand in enumerate(candidates, 1):
                prompt += f"{cand}\n"
            prompt += "\n"

        prompt += "Now select your action from belows within an <answer>...</answer> block:"
        return prompt


    def decide_action(self, prompt, max_retries=10) -> str:
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

            # Try to extract <answer>...</answer>
            match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if match:
                action = match.group(1).strip()
                full_input = prompt + generated_text
                full_input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)

                with torch.no_grad():
                    outputs = self.model.model(input_ids=full_input_ids)
                    logits = outputs.logits[:, :-1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                generated_ids = self.tokenizer(generated_text, return_tensors="pt").input_ids[:, 1:].to(self.device)
                selected_logprobs = log_probs[:, -generated_ids.size(1):].gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
                log_prob = selected_logprobs.sum(dim=-1).item()

                return action, log_prob
            
            print(f"[ActionAgent] <answer> not found, retrying ({attempt+1}/{max_retries})...")

        return generated_text.strip(), 0.0

    def update(self, prompt, action, old_logprobs, advantage, opt, clip_eps):
        inputs = self.tokenizer(prompt + "<answer>"+action+"</answer>", return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # new policy log_prob
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :].cpu()  # remove last token (predict next) [bs, seqlen, vocab]
        probs = torch.log_softmax(logits, dim=-1) 

        generate_ids = self.tokenizer(action, return_tensors="pt").input_ids[:, 1:]
        selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
        log_prob = selected_logprobs.sum(dim=-1)

        # PPO clipped loss
        ratio = torch.exp(log_prob - old_logprobs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

        action_loss = -torch.min(surr1, surr2).mean()
        if torch.isnan(action_loss) or torch.isinf(action_loss):
            print("Found NaN in action_loss, skipping step")
            return 0
        opt.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        opt.step()
        return action_loss
