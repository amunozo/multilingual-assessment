#!/usr/bin/env python3
"""Evaluate constituency predictions by span length and nonterminal."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path


def get_tree_spans(tree, root: bool, ignore_non_terminal: bool = False):
    spans = []
    if not isinstance(tree, str):
        if not root:
            if len(tree.leaves()) == 1:
                if "@" in tree.label():
                    spans = [(tuple(tree.leaves()), tree.label())]
            else:
                spans = [
                    (tuple(tree.leaves()), "-" if ignore_non_terminal else tree.label())
                ]
        for child in tree:
            if not isinstance(child, str):
                spans.extend(get_tree_spans(child, False, ignore_non_terminal))
    return spans


def _tree_spans(text: str):
    try:
        from nltk.tree import Tree
    except ImportError as exc:
        raise RuntimeError("Span evaluation requires NLTK.") from exc
    tree = Tree.fromstring(text, remove_empty_top_bracketing=True)
    tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")
    return get_tree_spans(tree, True)


def _validate_lengths(predicted_trees, gold_trees):
    if len(predicted_trees) != len(gold_trees):
        raise ValueError(
            f"Predicted {len(predicted_trees)} trees for {len(gold_trees)} gold trees."
        )


def performance_on_non_terminals(predicted_trees, gold_trees):
    _validate_lengths(predicted_trees, gold_trees)
    counts = {}
    for predicted, gold in zip(predicted_trees, gold_trees):
        predicted_spans = _tree_spans(predicted)
        remaining = copy.deepcopy(predicted_spans)
        for span in _tree_spans(gold):
            nonterminal = span[1].split("@", 1)[0]
            counts.setdefault(nonterminal, {"tp": 0.0, "fn": 0.0, "fp": 0.0})
            if span in remaining:
                counts[nonterminal]["tp"] += 1
                remaining.remove(span)
            else:
                counts[nonterminal]["fn"] += 1
        for span in remaining:
            nonterminal = span[1].split("@", 1)[0]
            counts.setdefault(nonterminal, {"tp": 0.0, "fn": 0.0, "fp": 0.0})
            counts[nonterminal]["fp"] += 1
    return _precision_recall(counts)


def performance_on_span_len(predicted_trees, gold_trees):
    _validate_lengths(predicted_trees, gold_trees)
    counts = {}
    for predicted, gold in zip(predicted_trees, gold_trees):
        remaining = copy.deepcopy(_tree_spans(predicted))
        for span in _tree_spans(gold):
            length = len(span[0])
            counts.setdefault(length, {"tp": 0.0, "fn": 0.0, "fp": 0.0})
            if span in remaining:
                counts[length]["tp"] += 1
                remaining.remove(span)
            else:
                counts[length]["fn"] += 1
        for span in remaining:
            length = len(span[0])
            counts.setdefault(length, {"tp": 0.0, "fn": 0.0, "fp": 0.0})
            counts[length]["fp"] += 1
    return _precision_recall(counts)


def _precision_recall(counts):
    result = {}
    for key, values in counts.items():
        precision_denominator = values["tp"] + values["fp"]
        recall_denominator = values["tp"] + values["fn"]
        result[key] = {
            "p": values["tp"] / precision_denominator if precision_denominator else 0.0,
            "r": values["tp"] / recall_denominator if recall_denominator else 0.0,
        }
    return result


def average_nonterminal_lengths(path: Path | str):
    nonterminals = []
    lengths: dict[str, list[int]] = {}
    for tree in Path(path).read_text(encoding="utf-8").splitlines():
        for leaves, label in _tree_spans(tree):
            nonterminal = label.split("@", 1)[0]
            nonterminals.append(nonterminal)
            lengths.setdefault(nonterminal, []).append(len(leaves))
    return Counter(nonterminals), {
        key: sum(values) / len(values) for key, values in lengths.items()
    }


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--span-length-threshold", type=int, default=25)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Plotting requires pandas, matplotlib, and seaborn.") from exc
    gold = Path(args.gold).read_text(encoding="utf-8").splitlines()
    rows = []
    for predicted_file in sorted(path for path in args.predicted.iterdir() if path.is_file()):
        predicted = predicted_file.read_text(encoding="utf-8").splitlines()
        scores = performance_on_span_len(predicted, gold)
        for length, values in scores.items():
            if length <= args.span_length_threshold:
                rows.append(
                    {
                        "span length": length,
                        "F1": f1(values["p"], values["r"]),
                        "model": predicted_file.name,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=pd.DataFrame(rows),
        x="span length",
        y="F1",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=axis,
    )
    axis.set_ylim(0, 1)
    figure.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
