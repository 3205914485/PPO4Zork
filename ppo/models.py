from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
import os

class ActorModel(nn.Module):
    def __init__(self, base_model_name_or_path, lora_path, device):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True, torch_dtype=torch.float32, device_map=device)
        self.model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float32, device_map=device)
        for name, param in self.model.named_parameters(): ##  
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
        print(self.model.print_trainable_parameters())
        # self.model.to(device)
        self.device = device

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class CriticModel(nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.model = base_model
        hidden_size = base_model.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch.float32).to(device) # float16 will cause nan
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.model.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_idx = attention_mask.sum(1) - 1
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]
        last_hidden = last_hidden.to(self.value_head.weight.dtype)
        score = self.value_head(last_hidden).squeeze(-1)
        return score 


class RewardModelWithLoRA(nn.Module):
    def __init__(self, base_model_name_or_path, lora_path, device):
        super().__init__()
       
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        self.device = device
        self.model = PeftModel.from_pretrained(base_model, lora_path).to(device)

        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        
        value_head_path = os.path.join(lora_path, "value_head.pt")
        self.value_head.load_state_dict(torch.load(value_head_path, map_location=device))

    def forward(self, input_ids, attention_mask):
        
        outputs = self.model.base_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_token_idx = attention_mask.sum(1) - 1
        final_hidden = last_hidden[torch.arange(last_hidden.size(0)), last_token_idx]
        reward = self.value_head(final_hidden).squeeze(-1)
        return reward

class RewardModel(nn.Module):
    def __init__(self, base_model_name, device):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        self.device = device
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        last_token_idx = attention_mask.sum(1) - 1  # [batch]
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]  # [batch, hidden]
        rewards = self.value_head(last_hidden).squeeze(-1)  # [batch]
        return rewards

class CriticModel_RM(nn.Module):
    def __init__(self, base_model_name, device):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_idx = attention_mask.sum(1) - 1
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]
        rewards = self.value_head(last_hidden).squeeze(-1)
        return rewards

def load_models(args):

    if args.rm_use:
        if args.rm_lora:
            RM = RewardModelWithLoRA(args.rm_path, args.rm_lora_path, args.rm_device)
            RM.eval()
        else :
            RM = RewardModel(args.rm_path, device=args.rm_device)
            RM.load_state_dict(torch.load(args.rm_ckpts, map_location=args.rm_device))
            RM.to(args.rm_device)
            RM.eval()
    else:
        RM = None

    AM = ActorModel(args.am_path, args.am_lora_path, device=args.am_device)
    
    if args.cm_use_rm:
        CM = CriticModel_RM(args.rm_path, device=args.cm_device)
        CM.load_state_dict(torch.load(args.rm_ckpts, map_location=args.cm_device))
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=10,
            lora_alpha=16,
            lora_dropout=0.05
        )
        CM.model = get_peft_model(CM.model, peft_config)
        CM.to(args.cm_device)
        print(CM.model.print_trainable_parameters())
    else:
        CM = CriticModel(AM, device=args.cm_device)

    return AM, CM, RM


