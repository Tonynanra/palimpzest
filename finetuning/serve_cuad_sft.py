"""Launch a Qwen CUAD adapter through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .cuad_sft_config import load_run_config
except ImportError:
    from cuad_sft_config import load_run_config


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
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a CUAD Qwen adapter with vLLM")
    parser.add_argument("--run-dir", type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument("--adapter-dir", type=Path, default=None)
    artifact.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-lora-rank", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    config = load_run_config(args.run_dir)
    args.model_id = config.model_id
    if args.merged_dir is None:
        args.adapter_dir = args.adapter_dir or args.run_dir / "adapter"
    default_name = "cuad-" + config.model_id.split("/")[-1].lower().replace(".", "")
    args.adapter_name = args.adapter_name or default_name
    args.served_model_name = args.served_model_name or default_name
    if args.max_lora_rank is None:
        args.max_lora_rank = config.lora_rank
    return args


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
