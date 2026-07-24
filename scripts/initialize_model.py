#!/usr/bin/env python3
"""Create a randomly initialized token-classification checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model ID providing the architecture config.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before initializing a model.") from exc

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, num_labels=args.num_labels)
    model = AutoModelForTokenClassification.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
