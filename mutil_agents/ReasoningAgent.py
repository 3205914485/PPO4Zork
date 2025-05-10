import torch

class ReasoningAgent:
    def __init__(self, model, tokenizer, device='cuda:0', max_tokens=128):
        self.model = model  # ActorModel, 需要有 .model 属性
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def build_prompt(self, observation: str, memory_summary: str) -> str:
        return (
            "You are playing the text-based game Zork.\n"
            "Below is your current situation and known memory. Think carefully and explain what is happening and what to consider next.\n\n"
            f"--- Observation ---\n{observation.strip()}\n\n"
            f"--- Memory ---\n{memory_summary.strip()}\n\n"
        )

    def generate_thought(self, prompt: str) -> str:
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
            text = self.tokenizer.decode(generated, skip_special_tokens=True)

        full_input = prompt + text
        full_input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.model(input_ids=full_input_ids)
        logits = outputs.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        generated_ids = self.tokenizer(text, return_tensors="pt").input_ids[:, 1:].to(self.device)
        selected_logprobs = log_probs[:, -generated_ids.size(1):].gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
        log_prob = selected_logprobs.sum(dim=-1).item()

        return text.strip(), log_prob
    
    def update(self, prompt, thought, old_logprobs, advantage, opt, clip_eps):
        inputs = self.tokenizer(prompt + thought, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # new policy log_prob
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :].cpu()  # remove last token (predict next) [bs, seqlen, vocab]
        probs = torch.log_softmax(logits, dim=-1) 

        generate_ids = self.tokenizer(thought, return_tensors="pt").input_ids[:, 1:]
        selected_logprobs = probs[:, -generate_ids.size(1):].gather(-1, generate_ids.unsqueeze(-1)).squeeze(-1)
        log_prob = selected_logprobs.sum(dim=-1)

        # PPO clipped loss
        ratio = torch.exp(log_prob - old_logprobs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

        thought_loss = -torch.min(surr1, surr2).mean()
        if torch.isnan(thought_loss) or torch.isinf(thought_loss):
            print("Found NaN in thought_loss, skipping step")
            return 0
        opt.zero_grad()
        thought_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        opt.step()
        return thought_loss
