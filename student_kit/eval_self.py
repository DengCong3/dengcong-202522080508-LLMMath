"""Self-evaluation script: base model vs fine-tuned adapter on valid.jsonl.

Compares two configurations:
- Base Gemma 3 270M (no adapter)
- Fine-tuned (QLoRA / LoRA / manual merge)

Scores outputs with student_kit/reward.py and writes results.json.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

# ---------------------------------------------------------------------------
# Stage 0: Environment check
# ---------------------------------------------------------------------------
print("=" * 60)
print("ENVIRONMENT CHECK (eval_self)")

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    print(f"  torch          : {torch.__version__} (CUDA: {torch.version.cuda})")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
else:
    print(f"  torch          : {torch.__version__} (CPU only)")

try:
    import transformers
    print(f"  transformers   : {transformers.__version__}")
except ImportError:
    print("  transformers   : MISSING")
    raise

try:
    import peft
    print(f"  peft           : {peft.__version__} (available)")
    _peft_available = True
except ImportError:
    print("  peft           : not installed (using manual merge)")
    _peft_available = False

print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import compute_reward


# ---------------------------------------------------------------------------
# Stage 1: Manual LoRA (fallback when PEFT is unavailable)
# ---------------------------------------------------------------------------

class _ManualLoraLinear(torch.nn.Module):
    """LoRA-adapted linear layer for inference-time weight merge."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_alpha: float,
        rank: int,
    ):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora_A = torch.nn.Parameter(lora_A.contiguous())
        self.lora_B = torch.nn.Parameter(lora_B.contiguous())
        self.scaling = lora_alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling
        return base + lora


def _manual_lora_merge(model: torch.nn.Module, adapter_dir: Path, rank: int = 8, lora_alpha: float = 16) -> torch.nn.Module:
    state_dict = torch.load(adapter_dir / "adapter_model.safetensors", map_location="cpu")
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lora_A_key = f"{name}.lora_A"
        lora_B_key = f"{name}.lora_B"
        if lora_A_key not in state_dict or lora_B_key not in state_dict:
            continue
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        lora_linear = _ManualLoraLinear(
            module,
            state_dict[lora_A_key],
            state_dict[lora_B_key],
            lora_alpha=lora_alpha,
            rank=rank,
        )
        lora_linear.to(next(module.parameters()).device)
        setattr(parent, attr_name, lora_linear)
        replaced += 1
    print(f"  Manual LoRA merge: {replaced} layers replaced")
    return model


def load_finetuned_model(model_dir: str, adapter_dir: str, device: str = "auto"):
    """Load base model + fine-tuned adapter, trying PEFT first, then manual."""
    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    rank = 8
    lora_alpha = 16
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        rank = cfg.get("r", 8)
        lora_alpha = cfg.get("lora_alpha", 16)

    if _peft_available:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_path)
        print("  Loaded with PeftModel (PEFT)")
        return model
    print("  PeftModel unavailable; falling back to manual LoRA merge")
    return _manual_lora_merge(base, adapter_path, rank=rank, lora_alpha=lora_alpha)


# ---------------------------------------------------------------------------
# Stage 2: Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert logo designer working in clean, scalable vector graphics. "
    "Given a description of a logo's visual elements, output ONE complete SVG document for the logo.\n\n"
    "Rules:\n"
    "- Output ONLY the SVG: a single <svg ...>...</svg> element with an xmlns and viewBox=\"0 0 256 256\". "
    "No prose, no markdown, no code fences.\n"
    "- Compose centered, content roughly within 16..240. Use a small cohesive palette.\n"
    "- Put gradients/filters in <defs>; use vector primitives only "
    "(<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, <g>). No <image>, external refs, or scripts.\n"
    "- Draw exactly what the description specifies."
)


def extract_user_prompt(example: Dict[str, Any]) -> str:
    msgs = example.get("messages", [])
    return next((m["content"] for m in msgs if m.get("role") == "user"), "")


def build_chat_prompt(user: str, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _clean_generated_text(text: str) -> str:
    """Strip obvious non-SVG wrappers and trailing artifacts."""
    # remove common leakage prefixes
    for prefix in [
        "You are a professional",
        "You are an expert",
        "Requirements:",
        "Rules:",
        "A soft rounded-square badge",
    ]:
        if text.startswith(prefix):
            idx = text.find("<svg")
            if idx != -1:
                text = text[idx:]
                break
    # cut off obvious trailing artifacts
    for stop in ["<eos>", "<|eot_id|>", "<|end_of_text|>"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    stop_token_ids = []
    for token in ["</svg>", "<eos>", "\n\n\n", "<|eot_id|>", "<|end_of_text|>"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            stop_token_ids.extend(ids)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # take only newly generated tokens when possible
    if out.shape[1] > prompt_length:
        new_tokens = out[:, prompt_length:]
        decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    else:
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
    decoded = _clean_generated_text(decoded)
    return decoded.strip()


def load_examples(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Stage 3: Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(model, tokenizer, examples: List[Dict[str, Any]], max_new_tokens: int):
    from tqdm import tqdm
    rows: List[Dict[str, Any]] = []
    for ex in tqdm(examples, desc="Generating"):
        prompt_text = extract_user_prompt(ex)
        chat_prompt = build_chat_prompt(prompt_text, tokenizer)
        text = generate(model, tokenizer, chat_prompt, max_new_tokens=max_new_tokens)
        reward_info = compute_reward(text, prompt=prompt_text)
        rows.append({
            "prompt": prompt_text,
            "text": text,
            "reward": reward_info["reward"],
            "ok": reward_info["ok"],
            "subscores": reward_info["subscores"],
            "details": reward_info["details"],
        })
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [r["reward"] for r in rows]
    ok_rate = sum(1.0 for r in rows if r["ok"]) / max(len(rows), 1)
    return {
        "count": len(rows),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "valid_rate": ok_rate,
    }


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _delta(b_value: float, f_value: float) -> float:
    return f_value - b_value


def pretty_print_results(results: Dict[str, Any]) -> None:
    if "base" not in results or "fine_tuned" not in results:
        return
    base_agg = results["base"]["aggregate"]
    ft_agg = results["fine_tuned"]["aggregate"]
    b_rows = results["base"]["rows"]
    f_rows = results["fine_tuned"]["rows"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<24} {'Base':>10} {'Finetune':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Mean Reward':<24} {base_agg['mean_reward']:>10.4f} {ft_agg['mean_reward']:>10.4f} {_delta(base_agg['mean_reward'], ft_agg['mean_reward']):>+10.4f}")
    print(f"{'Valid Rate':<24} {base_agg['valid_rate']:>10.4f} {ft_agg['valid_rate']:>10.4f} {_delta(base_agg['valid_rate'], ft_agg['valid_rate']):>+10.4f}")

    subscore_keys = [
        "valid_structure",
        "clean_extraction",
        "length",
        "palette",
        "coordinates",
        "prompt_coverage",
        "element_diversity",
        "smoothness",
    ]
    for key in subscore_keys:
        b_vals = [r["subscores"].get(key, 0.0) for r in b_rows]
        f_vals = [r["subscores"].get(key, 0.0) for r in f_rows]
        print(f"{key:<24} {_mean(b_vals):>10.4f} {_mean(f_vals):>10.4f} {_delta(_mean(b_vals), _mean(f_vals)):>+10.4f}")

    print("=" * 60)
    print("Best / Worst samples by reward delta")
    print("=" * 60)
    deltas = []
    for i, (b, f) in enumerate(zip(b_rows, f_rows)):
        deltas.append((_delta(b["reward"], f["reward"]), i, b, f))
    deltas.sort(reverse=True)
    for rank, entry in enumerate(deltas[:3], 1):
        _, idx, b, f = entry
        print(f"TOP{rank} idx={idx} delta={entry[0]:+.4f}")
        print(f"  prompt: {b['prompt'][:100]}...")
        print(f"  base:   {b['text'][:120].replace(chr(10), ' ')}...")
        print(f"  finetune: {f['text'][:120].replace(chr(10), ' ')}...")
    print()


# ---------------------------------------------------------------------------
# Stage 4: Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="./gemma3-270m")
    p.add_argument("--adapter_dir", type=str, default="./adapter")
    p.add_argument("--valid_file", type=str, default="./dataset/logo-detailed-prompt/valid.jsonl")
    p.add_argument("--output", type=str, default="./student_kit/results.json")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--skip_base", action="store_true")
    p.add_argument("--skip_finetuned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.valid_file)
    print(f"\nLoaded {len(examples)} validation examples\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results: Dict[str, Any] = {"base": {}, "fine_tuned": {}}

    if not args.skip_base:
        print("=" * 50)
        print("Evaluating BASE model")
        print("=" * 50)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map=args.device,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        base_rows = evaluate_split(base_model, tokenizer, examples, args.max_new_tokens)
        results["base"] = {"aggregate": aggregate(base_rows), "rows": base_rows}
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    if not args.skip_finetuned:
        print("=" * 50)
        print("Evaluating FINE-TUNED model")
        print("=" * 50)
        ft_model = load_finetuned_model(args.model_dir, args.adapter_dir, args.device)
        ft_model.eval()
        ft_rows = evaluate_split(ft_model, tokenizer, examples, args.max_new_tokens)
        results["fine_tuned"] = {"aggregate": aggregate(ft_rows), "rows": ft_rows}
        del ft_model
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    if "base" in results and "fine_tuned" in results:
        br = results["base"]["aggregate"]["mean_reward"]
        fr = results["fine_tuned"]["aggregate"]["mean_reward"]
        print(f"\nBase     reward: {br:.4f}")
        print(f"Finetune reward: {fr:.4f}")
        print(f"Delta:          {fr - br:+.4f}")
        pretty_print_results(results)


if __name__ == "__main__":
    main()
