import os
import jsonlines
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

BASE_DIR = "/workspace/svg_logo_task"
BASE_MODEL = os.path.join(BASE_DIR, "gemma3-270m")
TRAIN_JSONL = os.path.join(BASE_DIR, "dataset", "logo-detailed-prompt", "train.jsonl")
OUTPUT_ADAPTER = os.path.join(BASE_DIR, "adapter")
os.environ["PYTHONPATH"] = BASE_DIR

gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
MASK_IGNORE = -100

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto"
)
model = get_peft_model(model, lora_config)
print("===== 可训练参数 =====")
model.print_trainable_parameters()

def format_gemma(msgs):
    text = ""
    for m in msgs:
        text += f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
    return text

# 仅保存原始消息字符串，实时编码
class SvgTrainDataset(Dataset):
    def __init__(self, jsonl_path, max_len=1024):
        self.max_len = max_len
        self.raw_msgs = []
        with jsonlines.open(jsonl_path, "r") as reader:
            for item in reader:
                self.raw_msgs.append(item["messages"])

    def __len__(self):
        return len(self.raw_msgs)

    def __getitem__(self, idx):
        messages = self.raw_msgs[idx]
        full_str = format_gemma(messages)
        prompt_str = format_gemma(messages[:-1])

        full_tok = tokenizer(full_str, truncation=True, max_length=self.max_len, padding="max_length")
        prompt_tok = tokenizer(prompt_str, truncation=True, max_length=self.max_len)

        labels = full_tok["input_ids"].copy()
        prompt_len = len(prompt_tok["input_ids"])
        for i in range(prompt_len):
            labels[i] = MASK_IGNORE
        full_tok["labels"] = labels
        return full_tok

train_dataset = SvgTrainDataset(TRAIN_JSONL, max_len=1024)
print(f"总训练样本数: {len(train_dataset)}")

# 单次只跑1个epoch，降低单次运行内存累积量
train_args = TrainingArguments(
    output_dir=OUTPUT_ADAPTER,
    learning_rate=3e-4,
    num_train_epochs=1,  # 单次仅1轮
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    bf16=False,
    fp16=False,
    save_steps=10,
    logging_steps=5,
    save_total_limit=2,
    report_to="none",
    resume_from_checkpoint=True,  # 自动加载上一轮权重续训
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset)

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(OUTPUT_ADAPTER)
    print(f"本轮Epoch完成，LoRA已保存至 {OUTPUT_ADAPTER}")
