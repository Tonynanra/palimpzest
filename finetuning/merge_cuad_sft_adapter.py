"""Merge a CUAD PEFT adapter into a standalone Qwen text checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .cuad_sft_config import load_run_config
except ImportError:
    from cuad_sft_config import load_run_config


def merge_adapter(model_id: str, adapter_dir: Path, output_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a CUAD Qwen LoRA adapter")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    config = load_run_config(args.run_dir)
    args.model_id = config.model_id
    args.adapter_dir = args.adapter_dir or args.run_dir / "adapter"
    args.output_dir = args.output_dir or args.run_dir / "merged"
    return args


def main() -> None:
    args = parse_args()
    print(merge_adapter(args.model_id, args.adapter_dir, args.output_dir))


if __name__ == "__main__":
    main()
