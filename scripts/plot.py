#!/usr/bin/env python3
"""Plot the relative error reduction from pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path


def relative_error_reduction(pretrained_score: float, random_score: float) -> float:
    """Return percentage error reduction over a randomly initialized model."""
    baseline_error = 100.0 - random_score
    if baseline_error == 0:
        return 0.0
    return ((100.0 - random_score) - (100.0 - pretrained_score)) / baseline_error * 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="formalism", required=True)
    for name in ("dependency", "constituency"):
        command = subparsers.add_parser(name)
        command.add_argument("--scores", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--encoding")
    return parser


def _prepare_dependency(frame, encoding):
    frame = frame[frame["Finetuned"] == "not_finetuned"].drop_duplicates()
    if encoding:
        frame = frame[frame["Encoding"] == encoding]
    index = ["Treebank", "Language Model", "Encoding"]
    pivot = frame.pivot_table(index=index, columns="Pretrained", values="LAS", aggfunc="mean")
    required = {"pretrained", "not_pretrained"}
    if not required.issubset(pivot.columns):
        raise ValueError("Scores must contain pretrained and not_pretrained LAS rows.")
    pivot = pivot.reset_index()
    pivot["Relative error reduction"] = pivot.apply(
        lambda row: relative_error_reduction(row["pretrained"], row["not_pretrained"]),
        axis=1,
    )
    return pivot, "Treebank"


def _prepare_constituency(frame, encoding):
    frame = frame[frame["finetuned"] == "not_finetuned"].drop_duplicates()
    if encoding:
        frame = frame[frame["encoding"] == encoding]
    index = ["language", "lm", "encoding"]
    pivot = frame.pivot_table(
        index=index, columns="pretrained", values="BracketingFMeasure", aggfunc="mean"
    )
    required = {"pretrained", "not_pretrained"}
    if not required.issubset(pivot.columns):
        raise ValueError(
            "Scores must contain pretrained and not_pretrained BracketingFMeasure rows."
        )
    pivot = pivot.reset_index()
    pivot["Relative error reduction"] = pivot.apply(
        lambda row: relative_error_reduction(row["pretrained"], row["not_pretrained"]),
        axis=1,
    )
    pivot = pivot.rename(columns={"lm": "Language Model"})
    return pivot, "language"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Plotting requires pandas, matplotlib, and seaborn.") from exc

    frame = pd.read_csv(args.scores)
    if args.formalism == "dependency":
        plot_data, x_column = _prepare_dependency(frame, args.encoding)
    else:
        plot_data, x_column = _prepare_constituency(frame, args.encoding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    sns.barplot(
        data=plot_data,
        x=x_column,
        y="Relative error reduction",
        hue="Language Model",
        palette="Set2",
        ax=axis,
    )
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
