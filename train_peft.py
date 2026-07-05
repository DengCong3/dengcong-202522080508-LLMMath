import os
# ms-swift标准API导入
from swift import sft_main, Sft

# 固定项目路径
os.environ["PYTHONPATH"] = "."

def run_train():
    args = Sft(
        model="./gemma3-270m",
        model_type="gemma3_text",
        dataset="./train.jsonl",
        train_dataset="./train.jsonl",
        val_dataset="./valid.jsonl",
        dataset_type="chatml",
        mask_prompt=True,
        lora_rank=16,
        lora_alpha=32,
        learning_rate=3e-4,
        num_train_epochs=3,
        max_seq_len=2048,
        max_new_tokens=1800,
        reward_module="student_kit.reward:calculate_reward",
        rl_type=None,
        output_dir="./adapter",
        save_steps=200,
        eval_steps=300,
        val_size=0.1,
        gradient_checkpointing=True,
        use_flash_attn=False,
        per_device_train_batch_size=1,
        load_in_4bit=False,
        bf16=False
    )
    # 启动训练
    sft_main(args)

if __name__ == "__main__":
    run_train()
