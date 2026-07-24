#!/usr/bin/env python3
"""Predict and evaluate one completed probing experiment."""

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
    dependency = subparsers.add_parser("dependency", help="Evaluate a UD treebank.")
    dependency.add_argument("--treebank", required=True)
    dependency.add_argument("--ud-root", required=True, type=Path)
    dependency.add_argument("--encoding", required=True)

    constituency = subparsers.add_parser(
        "constituency", help="Evaluate PTB, CTB, or one SPMRL treebank."
    )
    constituency.add_argument("--language", required=True)
    constituency.add_argument("--corpus-root", required=True, type=Path)
    constituency.add_argument("--encoding", default="const")

    for command in (dependency, constituency):
        command.add_argument("--model", required=True)
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
        command.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
        command.add_argument("--artifact-root", type=Path)
        command.add_argument("--scores", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = {}
    if args.artifact_root is not None:
        options["artifact_root"] = args.artifact_root
    if args.scores is not None:
        options["scores_path"] = args.scores
    common = {
        "lm": args.model,
        "finetuned": args.finetuned,
        "pretrained": args.pretrained,
        "encoding": args.encoding,
        "task": args.task,
    }
    predict_options = {"device": args.device}
    if args.artifact_root is not None:
        predict_options["artifact_root"] = args.artifact_root
    if args.formalism == "dependency":
        from src import dep

        dep.predict(args.treebank, **common, **predict_options)
        result = dep.evaluate(args.treebank, ud_root=args.ud_root, **common, **options)
    else:
        from src import const

        const.predict(args.language, **common, **predict_options)
        result = const.evaluate(
            args.language, corpus_root=args.corpus_root, **common, **options
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
