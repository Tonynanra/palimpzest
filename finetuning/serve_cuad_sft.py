"""Launch a Qwen3.8-27B-FP8 CUAD adapter through vLLM."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .train_cuad_sft import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_MODEL_ID
except ImportError:
    from train_cuad_sft import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_MODEL_ID


def build_command(args: argparse.Namespace) -> list[str]:
    model = str(args.merged_dir) if args.merged_dir else args.model_id

    command = [
        "vllm",
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "auto",
        "--max-model-len",
        str(args.max_model_len),
        "--language-model-only",
        "--served-model-name",
        args.served_model_name,
    ]
    if not args.merged_dir:
        command.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{args.adapter_name}={args.adapter_dir}",
                "--max-lora-rank",
                str(args.max_lora_rank),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a CUAD Qwen3.8-27B-FP8 adapter with vLLM")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--adapter-name", default="cuad-qwen38-27b-fp8")
    parser.add_argument("--served-model-name", default="cuad-qwen38-27b-fp8")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.adapter_dir is None) == (args.merged_dir is None):
        parser.error("provide exactly one of --adapter-dir or --merged-dir")
    command = build_command(args)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
