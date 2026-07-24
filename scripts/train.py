#!/usr/bin/env python3
"""Train one probing experiment with explicit data and output paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="formalism", required=True)

    dependency = subparsers.add_parser("dependency", help="Train on a UD treebank.")
    dependency.add_argument("--treebank", required=True)
    dependency.add_argument("--ud-root", required=True, type=Path)
    dependency.add_argument("--encoding", required=True)

    constituency = subparsers.add_parser(
        "constituency", help="Train on PTB, CTB, or one SPMRL treebank."
    )
    constituency.add_argument("--language", required=True)
    constituency.add_argument("--corpus-root", required=True, type=Path)
    constituency.add_argument("--encoding", default="const")

    for command in (dependency, constituency):
        command.add_argument("--model", required=True, help="Hugging Face model ID or local path.")
        command.add_argument(
            "--finetuned",
            choices=("finetuned", "not_finetuned"),
            default="not_finetuned",
        )
        command.add_argument(
            "--pretrained",
            choices=("pretrained", "not_pretrained"),
            default="pretrained",
        )
        command.add_argument("--task", choices=("single",), default="single")
        command.add_argument("--epochs", type=int, default=20)
        command.add_argument("--seed", type=int, default=13)
        command.add_argument("--not-ft-lr", type=float, default=2e-3)
        command.add_argument("--ft-lr", type=float, default=5e-5)
        command.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
        command.add_argument("--encoded-root", type=Path)
        command.add_argument("--artifact-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path_options = {}
    if args.encoded_root is not None:
        path_options["encoded_root"] = args.encoded_root
    if args.artifact_root is not None:
        path_options["artifact_root"] = args.artifact_root
    common = {
        "lm": args.model,
        "finetuned": args.finetuned,
        "pretrained": args.pretrained,
        "encoding": args.encoding,
        "task": args.task,
        "not_ft_lr": args.not_ft_lr,
        "ft_lr": args.ft_lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
        **path_options,
    }
    if args.formalism == "dependency":
        from src import dep

        output = dep.train(args.treebank, ud_root=args.ud_root, **common)
    else:
        from src import const

        output = const.train(args.language, corpus_root=args.corpus_root, **common)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
