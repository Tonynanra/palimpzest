"""Launch a Qwen3.5 CUAD adapter through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


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
        "bfloat16",
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
    parser = argparse.ArgumentParser(description="Serve a CUAD Qwen3.5 adapter with vLLM")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--adapter-name", default="cuad-qwen35-4b")
    parser.add_argument("--served-model-name", default="cuad-qwen35-4b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=32768)
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
