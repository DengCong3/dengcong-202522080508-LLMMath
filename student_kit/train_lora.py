"""LoRA fine-tuning script for Gemma 3 270M on SVG logo generation.

Framework: transformers + PEFT/LoRA (BF16 or 4-bit QLoRA)
Target hardware: NVIDIA V100 32GB

Usage
-----
python train_lora.py \
  --model_dir ./gemma3-270m \
  --train_file ./dataset/logo-detailed-prompt/train.jsonl \
  --valid_file ./dataset/logo-detailed-prompt/valid.jsonl \
  --output_dir ./adapter \
  --max_steps 2000 \
  --learning_rate 2e-4
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Stage 0: Environment diagnostics
# ---------------------------------------------------------------------------
print("=" * 60)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 60)

def diag(name: str, loader):
    try:
        mod = loader()
        v = getattr(mod, "__version__", "unknown")
        extra = getattr(mod, "__file__", "") or ""
        print(f"  {name:<22}: {v}  [{extra}]")
        return mod
    except ImportError as e:
        print(f"  {name:<22}: NOT INSTALLED ({e})")
        return None
    except Exception as e:
        print(f"  {name:<22}: ERROR ({e})")
        return None

torch_mod = diag("torch", lambda: __import__("torch"))
hf_mod = diag("transformers", lambda: __import__("transformers"))
bnb_mod = diag("bitsandbytes", lambda: __import__("bitsandbytes"))
peft_mod = diag("peft", lambda: __import__("peft"))
ds_mod = diag("datasets", lambda: __import__("datasets"))

import torch as _torch
print(f"  CUDA available      : {_torch.cuda.is_available()}")
if _torch.cuda.is_available():
    print(f"  CUDA version        : {_torch.version.cuda}")
    print(f"  GPU                 : {_torch.cuda.get_device_name(0)}")
    print(f"  GPU mem             : {_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"  python              : {sys.version.split()[0]}")
print("=" * 60)

torch = _torch

# ---------------------------------------------------------------------------
# Stage 1: Check what's usable
# ---------------------------------------------------------------------------
HAS_CUDA = _torch.cuda.is_available()
HAS_BNB = bnb_mod is not None and hasattr(bnb_mod, "bitsandbytes")
HAS_PEFT = peft_mod is not None
HAS_DATASETS = ds_mod is not None

_BNB_LOADABLE = False
if HAS_BNB:
    try:
        import bitsandbytes as _bnb
        _ = _bnb.lib
        _BNB_LOADABLE = True
    except Exception:
        print("[WARN] bitsandbytes imported but lib load failed — QLoRA disabled")

if HAS_CUDA and _BNB_LOADABLE:
    LOAD_MODE = "qlora"
    print("\n[OK] Loading strategy: QLoRA (4-bit NF4 + LoRA)")
elif HAS_CUDA:
    LOAD_MODE = "lora"
    print("\n[OK] Loading strategy: LoRA (BF16 + manual LoRA, no 4-bit)")
else:
    LOAD_MODE = "cpu"
    print("\n[WARN] Loading strategy: CPU mode (very slow, not recommended)")

# ---------------------------------------------------------------------------
# Stage 2: Import transformers
# ---------------------------------------------------------------------------
print("\n[Stage 2] Importing transformers...")
try:
    import transformers
    print(f"  transformers {transformers.__version__} OK")
except Exception as e:
    print(f"[FATAL] Cannot import transformers: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage 3: Import components
# ---------------------------------------------------------------------------
print("[Stage 3] Importing training components...")
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        DefaultDataCollator,
        EarlyStoppingCallback,
    )
    print("  Trainer, AutoModel, etc. OK")
except ImportError as e:
    print(f"[FATAL] Cannot import from transformers: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from datasets import load_dataset
    print("  datasets OK")
except ImportError:
    print("[FATAL] Cannot import datasets")
    sys.exit(1)

# PEFT (optional)
_LoraConfig = None
_get_peft_model = None
if peft_mod is not None:
    try:
        import peft as _peft
        from peft import LoraConfig as _LC, get_peft_model as _GPM
        _LoraConfig = _LC
        _get_peft_model = _GPM
        print(f"  PEFT {_peft.__version__} OK (will use for LoRA injection)")
    except Exception as e:
        print(f"  PEFT import failed ({e}) — will use manual LoRA")
        _LoraConfig = None
        _get_peft_model = None
else:
    print("  PEFT not installed — will use manual LoRA")

# ---------------------------------------------------------------------------
# Stage 4: Manual LoRA (fallback when PEFT is unavailable)
# ---------------------------------------------------------------------------

class ManualLoraLinear(torch.nn.Module):
    """Replaces a linear layer with LoRA-adapted version (inplace, no PEFT)."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank if rank > 0 else 0.0

        if lora_dropout > 0.0:
            self.lora_dropout = torch.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = torch.nn.Identity()

        torch.nn.init.normal_(torch.empty(rank, linear.in_features), std=0.01)
        self.lora_A = torch.nn.Parameter(
            torch.randn(rank, linear.in_features) * 0.01
        )
        self.lora_B = torch.nn.Parameter(torch.zeros(linear.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + lora_out


def inject_lora_manual(
    model: torch.nn.Module,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> torch.nn.Module:
    """Replace target linear layers with LoRA-adapted versions."""
    replaced = 0
    target_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(tm) for tm in target_modules):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        new_module = ManualLoraLinear(module, rank, lora_alpha, lora_dropout)
        new_module.to(next(module.parameters()).device)
        setattr(parent, attr_name, new_module)
        replaced += 1

    print(f"  Manual LoRA: replaced {replaced} / {len(list(model.modules()))} modules")
    return model


def freeze_non_lora(model: torch.nn.Module):
    """Freeze all parameters except those with 'lora_' prefix."""
    frozen = 0
    trainable = 0
    for p in model.parameters():
        name = getattr(p, "name", "")
        if "lora_" in name:
            p.requires_grad = True
            trainable += 1
        else:
            p.requires_grad = False
            frozen += 1
    print(f"  Froze {frozen} params, {trainable} trainable")


def print_trainable_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ---------------------------------------------------------------------------
# Stage 5: Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str, load_mode: str = "qlora") -> torch.nn.Module:
    """Load Gemma 3 270M in one of 3 modes."""
    print(f"\n[Model Loading] mode={load_mode}")
    bnb_config = None
    if load_mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  4-bit NF4 quantization enabled")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if getattr(model.config, "attn_implementation", None) == "eager":
        # keep as-is for Gemma 3
        pass
    print(f"  Model loaded on device map: auto")
    return model

# ---------------------------------------------------------------------------
# Stage 6: Chat template helpers
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


def _build_prompt(example: dict) -> tuple[str, str]:
    msgs = example.get("messages", [])
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    prompt = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt, assistant


def _tokenize(tokenizer, prompt: str, assistant: str, max_seq_length: int) -> dict:
    """Tokenize one example, keeping loss only on the assistant response.

    Priority:
    1. Always keep the full assistant response if it fits.
    2. If total length exceeds max_seq_length, truncate the prompt from the left,
       because the tail of the prompt (user request + recent context) matters most.
    """
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    a_ids = tokenizer(assistant, add_special_tokens=False).input_ids

    # Ensure EOS token at end of assistant if tokenizer has one and response is missing it
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and a_ids and a_ids[-1] != eos_id:
        a_ids = a_ids + [eos_id]

    total = len(p_ids) + len(a_ids)
    if total > max_seq_length:
        # Prefer to keep the assistant response; truncate prompt from the left
        keep_prompt = max(0, max_seq_length - len(a_ids))
        if keep_prompt < len(p_ids):
            p_ids = p_ids[-keep_prompt:] if keep_prompt > 0 else []
        dropped = total - max_seq_length
        warnings.warn(
            f"Truncated {dropped} tokens from prompt to fit max_seq_length={max_seq_length}. "
            f"Consider increasing --max_seq_length if SVGs are often cut off.",
            stacklevel=2,
        )

    input_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids

    # Guard: if assistant itself is longer than max_seq_length,
    # truncate from the right so fixed-length padding remains valid.
    if len(input_ids) > max_seq_length:
        a_ids = a_ids[: max_seq_length - len(p_ids)]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids

    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_datasets(train_file: str, valid_file: str, tokenizer, max_seq_length: int):
    pad_id = tokenizer.pad_token_id
    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_valid = load_dataset("json", data_files=valid_file, split="train")

    def _preprocess(example: dict) -> dict:
        prompt, assistant = _build_prompt(example)
        enc = _tokenize(tokenizer, prompt, assistant, max_seq_length)
        length = enc["length"]
        return {
            "input_ids": enc["input_ids"] + [pad_id] * (max_seq_length - length),
            "labels": enc["labels"] + [-100] * (max_seq_length - length),
            "attention_mask": [1] * length + [0] * (max_seq_length - length),
        }

    train_ds = raw_train.map(
        _preprocess,
        remove_columns=raw_train.column_names,
        desc="Tokenizing train",
    )
    train_ds.set_format("torch", columns=["input_ids", "labels", "attention_mask"])

    valid_ds = raw_valid.map(
        _preprocess,
        remove_columns=raw_valid.column_names,
        desc="Tokenizing valid",
    )
    valid_ds.set_format("torch", columns=["input_ids", "labels", "attention_mask"])
    return train_ds, valid_ds

# ---------------------------------------------------------------------------
# Stage 7: Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="./gemma3-270m")
    p.add_argument("--train_file", type=str,
                   default="./dataset/logo-detailed-prompt/train.jsonl")
    p.add_argument("--valid_file", type=str,
                   default="./dataset/logo-detailed-prompt/valid.jsonl")
    p.add_argument("--output_dir", type=str, default="./adapter")
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Max tokens per example. Increase if SVGs are often truncated.")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_mode", choices=["qlora", "lora", "none"], default="none",
                   help="Manually override loading mode (default: auto)")
    p.add_argument("--early_stopping_patience", type=int, default=5,
                   help="Stop if eval loss doesn't improve for N evals.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.force_mode != "none":
        global LOAD_MODE
        LOAD_MODE = args.force_mode

    print(f"\n{'='*60}")
    print(f"STEP 1/4: Loading tokenizer from {args.model_dir}")
    print('='*60)
    tokenizer = load_tokenizer(args.model_dir)

    print(f"\n{'='*60}")
    print(f"STEP 2/4: Building datasets")
    print('='*60)
    train_ds, valid_ds = build_datasets(
        args.train_file, args.valid_file, tokenizer, args.max_seq_length
    )
    print(f"  Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    print(f"\n{'='*60}")
    print(f"STEP 3/4: Loading model (mode={LOAD_MODE})")
    print('='*60)
    model = load_model(args.model_dir, load_mode=LOAD_MODE)

    print(f"\n{'='*60}")
    print(f"STEP 4/4: Injecting LoRA")
    print('='*60)

    if _LoraConfig is not None and _get_peft_model is not None:
        print("  Using PEFT LoraConfig + get_peft_model")
        lora_config = _LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = _get_peft_model(model, lora_config)
    else:
        print("  Using manual LoRA injection (no PEFT)")
        model = inject_lora_manual(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        freeze_non_lora(model)

    print_trainable_params(model)

    # Choose optimizer compatible with current loading mode
    if LOAD_MODE == "qlora" and _BNB_LOADABLE:
        optim_name = "paged_adamw_8bit"
    else:
        optim_name = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=False,
        optim=optim_name,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DefaultDataCollator(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ] if args.early_stopping_patience > 0 else None,
    )

    print(f"\n{'='*60}")
    print(f"TRAINING START")
    print('='*60)
    trainer.train()

    print(f"\nSaving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("ALL DONE!")


if __name__ == "__main__":
    main()
