"""Merge a CUAD PEFT adapter into a standalone Qwen3.5 text checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a CUAD Qwen3.5 LoRA adapter")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(merge_adapter(args.model_id, args.adapter_dir, args.output_dir))


if __name__ == "__main__":
    main()
