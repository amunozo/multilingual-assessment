#!/usr/bin/env python3
"""Evaluate dependency performance by displacement and relation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WordConll:
    index: int
    word: str
    head: int
    relation: str


def read_conllu(sentence: str) -> list[WordConll]:
    tree = [WordConll(0, "-ROOT-", 0, "root")]
    for line in sentence.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(f"Expected ten CoNLL-U fields, found {len(fields)}: {line!r}")
        if "." in fields[0] or "-" in fields[0]:
            continue
        tree.append(WordConll(int(fields[0]), fields[1], int(fields[6]), fields[7]))
    return tree


def dependency_displacements(tree: list[WordConll], unlabeled: bool = False):
    values = []
    for word in tree[1:]:
        if word.head == 0:
            continue
        distance = word.index - word.head
        values.append(distance if unlabeled else f"{distance}_{word.relation}")
    return values


def word_relations(tree: list[WordConll], unlabeled: bool = True):
    return [
        word.relation if unlabeled else f"{word.head}_{word.relation}"
        for word in tree[1:]
    ]


def elements_from_conllu(path: Path | str, unlabeled: bool):
    text = Path(path).read_text(encoding="utf-8")
    distances = []
    relations = []
    for sentence in text.split("\n\n"):
        if not sentence.strip():
            continue
        tree = read_conllu(sentence)
        distances.extend(dependency_displacements(tree, unlabeled))
        relations.extend(word_relations(tree, unlabeled))
    return distances, relations


def _label(item: str) -> str:
    return item.split("_", 1)[1]


def dependency_head_performance(gold_labels, predicted_labels):
    """Precision and recall by dependency relation, requiring the correct head."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted relations have different lengths.")
    counts: dict[str, dict[str, float]] = {}
    for gold, predicted in zip(gold_labels, predicted_labels):
        gold_relation = _label(gold)
        predicted_relation = _label(predicted)
        counts.setdefault(gold_relation, {"gold": 0.0, "predicted": 0.0, "correct": 0.0})
        counts.setdefault(
            predicted_relation, {"gold": 0.0, "predicted": 0.0, "correct": 0.0}
        )
        counts[gold_relation]["gold"] += 1
        counts[predicted_relation]["predicted"] += 1
        if gold == predicted:
            counts[gold_relation]["correct"] += 1
    return _scores(counts)


def displacement_labeled_performance(
    gold_distances, predicted_distances, *, minimum_frequency: int = 10
):
    """Precision and recall by distance, requiring relation and distance to match."""
    if len(gold_distances) != len(predicted_distances):
        raise ValueError("Gold and predicted displacements have different lengths.")
    frequencies = Counter(gold_distances)
    counts: dict[int, dict[str, float]] = {}
    for gold, predicted in zip(gold_distances, predicted_distances):
        if frequencies[gold] < minimum_frequency:
            continue
        gold_distance = int(gold.split("_", 1)[0])
        predicted_distance = int(predicted.split("_", 1)[0])
        counts.setdefault(gold_distance, {"gold": 0.0, "predicted": 0.0, "correct": 0.0})
        counts.setdefault(
            predicted_distance, {"gold": 0.0, "predicted": 0.0, "correct": 0.0}
        )
        counts[gold_distance]["gold"] += 1
        counts[predicted_distance]["predicted"] += 1
        if gold == predicted:
            counts[gold_distance]["correct"] += 1
    return _scores(counts)


def _scores(counts):
    result = {}
    for category, values in counts.items():
        precision = (
            values["correct"] / values["predicted"] if values["predicted"] else 0.0
        )
        recall = values["correct"] / values["gold"] if values["gold"] else 0.0
        result[category] = {"p": precision, "r": recall}
    return result


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def model_legend(name: str) -> str:
    mapping = {
        "bert-base-multilingual-cased": "mBERT",
        "xlm-roberta-base": "XLM-R",
        "canine-c": "CANINE-C",
        "canine-s": "CANINE-S",
    }
    return mapping.get(name, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-frequency", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Plotting requires pandas, matplotlib, and seaborn.") from exc

    gold_distances, gold_relations = elements_from_conllu(args.gold, args.unlabeled)
    relation_frequency = Counter(
        gold_relations if args.unlabeled else [_label(item) for item in gold_relations]
    )
    distance_rows = []
    relation_rows = []
    for predicted_file in sorted(path for path in args.predicted.iterdir() if path.is_file()):
        predicted_distances, predicted_relations = elements_from_conllu(
            predicted_file, args.unlabeled
        )
        model = model_legend(predicted_file.name)
        if args.unlabeled:
            relation_scores = _categorical_scores(gold_relations, predicted_relations)
            distance_scores = _categorical_scores(gold_distances, predicted_distances)
        else:
            relation_scores = dependency_head_performance(gold_relations, predicted_relations)
            distance_scores = displacement_labeled_performance(
                gold_distances,
                predicted_distances,
                minimum_frequency=args.minimum_frequency,
            )
        for distance, scores in distance_scores.items():
            if abs(int(distance)) <= 20:
                distance_rows.append(
                    {"distance": int(distance), "f1": f1(scores["p"], scores["r"]), "model": model}
                )
        for relation, _ in relation_frequency.most_common(7):
            scores = relation_scores.get(relation, {"p": 0.0, "r": 0.0})
            relation_rows.append(
                {"relation": relation, "f1": f1(scores["p"], scores["r"]), "model": model}
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=pd.DataFrame(distance_rows),
        x="distance",
        y="f1",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=axis,
    )
    axis.set(xlabel="Dependency displacement", ylabel="F1", ylim=(0, 1))
    figure.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(figure)

    relation_output = args.output.with_name(
        args.output.stem.replace("displacements", "relations") + args.output.suffix
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=pd.DataFrame(relation_rows), x="relation", y="f1", hue="model", ax=axis
    )
    axis.set(xlabel="Dependency relation", ylabel="F1", ylim=(0, 1))
    figure.savefig(relation_output, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return 0


def _categorical_scores(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Gold and predicted sequences have different lengths.")
    counts = {}
    for gold_item, predicted_item in zip(gold, predicted):
        counts.setdefault(gold_item, {"gold": 0.0, "predicted": 0.0, "correct": 0.0})
        counts.setdefault(
            predicted_item, {"gold": 0.0, "predicted": 0.0, "correct": 0.0}
        )
        counts[gold_item]["gold"] += 1
        counts[predicted_item]["predicted"] += 1
        if gold_item == predicted_item:
            counts[gold_item]["correct"] += 1
    return _scores(counts)


if __name__ == "__main__":
    raise SystemExit(main())
