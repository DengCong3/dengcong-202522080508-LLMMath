import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# 导入统一打分函数
from reward import calculate_reward, get_full_reward_detail

BASE_DIR = "/workspace/svg_logo_task"
BASE_MODEL_PATH = os.path.join(BASE_DIR, "gemma3-270m")
LORA_PATH = os.path.join(BASE_DIR, "adapter")
VAL_JSONL = os.path.join(BASE_DIR, "dataset/logo-detailed-prompt/valid.jsonl")
OUTPUT_JSON = os.path.join(BASE_DIR, "results.json")

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 均衡显存分配，防止单卡爆显存卡死
base_model_raw = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float16,
    device_map="balanced"
)
lora_tuned_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="balanced"
    ),
    LORA_PATH
)
lora_tuned_model.eval()
base_model_raw.eval()

def format_gemma(msgs):
    text = ""
    for m in msgs:
        text += f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
    return text

def single_infer(model, system_content, prompt_text):
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_text}
    ]
    input_text = format_gemma(chat)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    full_decode = tokenizer.decode(out[0], skip_special_tokens=True)
    # 修复：分割方式提取新增生成内容，不再粗暴replace
    if input_text in full_decode:
        svg_part = full_decode.split(input_text)[-1].strip()
    else:
        svg_part = full_decode.strip()
    return svg_part

# 加载验证集
val_data = []
with open(VAL_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        val_data.append(json.loads(line))

result_list = []
sum_base = 0.0
sum_lora = 0.0
sample_count = len(val_data)

for idx, item in enumerate(val_data):
    print(f"正在处理第 {idx+1}/{sample_count} 条样本")
    messages = item["messages"]
    sys_msg = messages[0]["content"]
    user_prompt = messages[1]["content"]

    # 基座推理+打分（含分项明细）
    out_base = single_infer(base_model_raw, sys_msg, user_prompt)
    base_detail = get_full_reward_detail(user_prompt, out_base)
    score_base_total = base_detail["total_reward"]
    sum_base += score_base_total

    # LoRA推理+打分（含分项明细）
    out_lora = single_infer(lora_tuned_model, sys_msg, user_prompt)
    lora_detail = get_full_reward_detail(user_prompt, out_lora)
    score_lora_total = lora_detail["total_reward"]
    sum_lora += score_lora_total

    result_list.append({
        "user_prompt": user_prompt,
        "base_output": out_base,
        "lora_output": out_lora,
        "base_score_detail": base_detail,
        "lora_score_detail": lora_detail,
        "base_total": score_base_total,
        "lora_total": score_lora_total
    })

# 统计均值
avg_base = sum_base / sample_count
avg_lora = sum_lora / sample_count
delta = avg_lora - avg_base

output_data = {
    "sample_num": sample_count,
    "stat": {
        "avg_base": round(avg_base,3),
        "avg_lora": round(avg_lora,3),
        "improve": round(delta,3)
    },
    "samples": result_list
}
with open(OUTPUT_JSON, "w", encoding="utf8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("评测全部完成！")
print(f"基座平均分：{avg_base:.3f}")
print(f"LoRA平均分：{avg_lora:.3f}")
print(f"分数提升：{delta:.3f}")