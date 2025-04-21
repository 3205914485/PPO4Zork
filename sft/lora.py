import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, LlamaForCausalLM, Qwen2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="qwen-2.5-3B-instruct", help="预训练模型的路径")
    parser.add_argument("--dataset_path", type=str, default="sft/data/processed/qwen-2.5-3B-instruct", help="处理后的数据集路径")
    parser.add_argument("--output_dir", type=str, default="sft/sft_model/qwen-2.5-3B_zork", help="模型保存的输出路径")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批量大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练的轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["WANDB_PROJECT"] = args.train_name # name your W&B project
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map=args.device)
    
    if isinstance(model, LlamaForCausalLM) or isinstance(model, Qwen2ForCausalLM):
        target_modules = ["q_proj",
                          "k_proj",
                          "v_proj",
                          "o_proj"]
        tokenizer.pad_token = tokenizer.eos_token
    else:
        target_modules = ["query_key_value"]

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA 的秩
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,  # 根据模型架构选择目标模块
    )

    # 将 LoRA 配置应用到模型
    model = get_peft_model(model, lora_config)

    # 加载已处理的数据集
    dataset = load_from_disk(args.dataset_path)
    dataset.set_format(
        type=dataset.format["type"],
        columns=list(dataset.features.keys()),
    )

    train_dataset = dataset

    def get_data_collator(tokenizer: AutoTokenizer):
        def data_collator(features: list) -> dict:
            len_ids = [len(feature["input_ids"]) for feature in features]
            longest = max(len_ids)
            input_ids = []
            labels_list = []
            for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
                ids = feature["input_ids"]
                seq_len = feature["seq_len"]
                labels = (
                    [-100] * (seq_len - 1) + ids[(seq_len - 1):] +
                    [-100] * (longest - ids_l)
                )
                ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
                _ids = torch.LongTensor(ids)
                labels_list.append(torch.LongTensor(labels))
                input_ids.append(_ids)
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels_list)
            return {
                "input_ids": input_ids,
                "labels": labels,
            }
        return data_collator

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_dir="sft/logs",
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,  # 使用混合精度训练
        dataloader_drop_last=True,
        remove_unused_columns=False
    )

    # 使用 Trainer 进行训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=get_data_collator(tokenizer)
    )

    # 开始训练
    trainer.train()

    print(f"Finished training for {args.epochs} epochs")

    # 保存微调后的模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model fine-tuned with LoRA saved to {args.output_dir}")

if __name__ == "__main__":
    main()
