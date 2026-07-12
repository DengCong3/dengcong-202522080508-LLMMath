"""Quick 10-example failure diagnosis for base vs finetune."""
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from train_lora import load_tokenizer  # keep shared tokenizer loader
from eval_self import build_chat_prompt, generate, load_finetuned_model
from reward import _extract_svg

BASE_MODEL_DIR = Path("./gemma3-270m")
ADAPTER_DIR = Path("./adapter")
VALID_FILE = Path("./dataset/logo-detailed-prompt/valid.jsonl")
N = 10


def count_tokens(text: str) -> int:
    return len(text.split())


def classify(text: str):
    text = text.strip()
    label = []

    # 1. still echoing system prompt / persona
    if re.search(r"You are (a professional|an expert)", text, re.IGNORECASE):
        label.append("echo_persona")

    # 2. svg structure
    svg, ok, details = _extract_svg(text)
    if ok:
        label.append("has_svg")
        if details.get("extra_text_before"):
            label.append("prefix_before_svg")
        if details.get("extra_text_after"):
            label.append("trailing_after_svg")
        if not re.search(r"xmlns\s*=\s*['\"]http://www\.w3\.org/2000/svg['\"]", svg, re.IGNORECASE):
            label.append("missing_xmlns")
        if not re.search(r"viewBox\s*=", svg, re.IGNORECASE):
            label.append("missing_viewbox")
        # 3. contains invalid fragments
        if re.search(r"<eos>|<g\s[^>]*fill=|url\(#\s*[A-Za-z]", text, re.IGNORECASE):
            label.append("fragmented_markup")
        if re.search(r"<[^>]{1,3}>", svg):
            label.append("short_invalid_tags")
    else:
        label.append("no_svg")

    # 4. runaway repetition / length
    if count_tokens(text) > 1200:
        label.append("very_long")
    if len(re.findall(r"vector brush", text, re.IGNORECASE)) > 3:
        label.append("repeated_phrase")

    return label, svg, details


def main():
    examples = []
    with VALID_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= N:
                break

    print(f"Loaded {len(examples)} examples from {VALID_FILE}")

    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_tok = load_tokenizer(BASE_MODEL_DIR)
    ft_model = load_finetuned_model(str(BASE_MODEL_DIR), adapter_dir=str(ADAPTER_DIR))
    ft_tok = load_tokenizer(BASE_MODEL_DIR)

    for i, ex in enumerate(examples, 1):
        user_text = next(m["content"] for m in ex.get("messages", []) if m.get("role") == "user")
        prompt = build_chat_prompt(user_text, ft_tok)

        base_out = generate(base_model, base_tok, prompt)
        ft_out = generate(ft_model, ft_tok, prompt)

        base_cls, base_svg, base_det = classify(base_out)
        ft_cls, ft_svg, ft_det = classify(ft_out)

        print("\n" + "=" * 70)
        print(f"Example {i:02d}")
        print("- Base labels:", ", ".join(base_cls) if base_cls else "ok")
        print("- FT   labels:", ", ".join(ft_cls) if ft_cls else "ok")
        print("- Base svg/viewbox/xmlns:", bool(base_svg), base_det.get("has_viewbox"), base_det.get("has_xmlns"))
        print("- FT   svg/viewbox/xmlns:", bool(ft_svg), ft_det.get("has_viewbox"), ft_det.get("has_xmlns"))
        print("- Base snippet:", base_out[:220].replace("\n", " ") + "...")
        print("- FT   snippet:", ft_out[:220].replace("\n", " ") + "...")
        if ft_det.get("trailing_text"):
            print("- FT trailing:", ft_det["trailing_text"].replace("\n", " "))


if __name__ == "__main__":
    main()
