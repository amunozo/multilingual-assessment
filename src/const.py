"""Constituency-parsing probe orchestration."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from . import canine, subword_models, util
from .paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ENCODED_ROOT,
    EXTERNAL_ROOT,
    PROJECT_ROOT,
    encoded_experiment_dir,
    model_dir,
    require_path,
)


def find_constituency_splits(
    language: str, corpus_root: Path | str
) -> dict[str, Path]:
    """Resolve the paper's PTB, CTB, or SPMRL split layout."""
    root = require_path(corpus_root, "constituency corpus root")
    normalized = language.lower()
    if normalized == "english":
        files = {
            "train": root / "train.trees",
            "validation": root / "dev.trees",
            "test": root / "test.trees",
        }
    elif normalized == "chinese":
        files = {
            "train": root / "train_ch.trees",
            "validation": root / "dev_ch.trees",
            "test": root / "test_ch.trees",
        }
    else:
        treebank = root / f"{language.upper()}_SPMRL" / "gold" / "ptb"
        display_name = language.capitalize()
        files = {
            "train": treebank / "train" / f"train.{display_name}.gold.ptb",
            "validation": treebank / "dev" / f"dev.{display_name}.gold.ptb",
            "test": treebank / "test" / f"test.{display_name}.gold.ptb",
        }
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing constituency split(s): " + ", ".join(absent))
    return files


def encode(
    language: str,
    task: str,
    output_dir: Path | str,
    *,
    corpus_root: Path | str,
) -> dict[str, Path]:
    """Encode constituency trees using the paper's relative-level encoding."""
    if task != "single":
        raise NotImplementedError("Only the single-task setup is supported by this pipeline.")
    try:
        from vendor.tree2labels_py3 import dataset as tree2labels_dataset
    except ImportError as exc:  # pragma: no cover - optional research stack
        raise RuntimeError(
            "Constituency encoding requires NumPy and NLTK; install requirements.txt."
        ) from exc

    source_files = find_constituency_splits(language, corpus_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, Path] = {}
    suffixes = {"train": "train", "validation": "dev", "test": "test"}
    for split, source in source_files.items():
        sequences, _ = tree2labels_dataset.transform_split(
            str(source),
            False,
            True,
            False,
            True,
            None,
            None,
            "~",
            "@",
        )
        destination = output / f"{language}-{suffixes[split]}.seq_lu"
        tree2labels_dataset.write_linearized_trees(str(destination), sequences, {})
        destinations[split] = destination
    return destinations


def _postprocess_labels(predictions: list[str]) -> list[str]:
    for index in range(1, len(predictions) - 2):
        if (
            "-BOS-" in predictions[index]
            or "-EOS-" in predictions[index]
            or predictions[index].startswith("NONE")
        ):
            predictions[index] = "1ROOT@S"
    if len(predictions) >= 2 and not predictions[-2].startswith("NONE"):
        predictions[-2] = "NONE"
    if predictions and predictions[-1] != "-EOS-":
        predictions[-1] = "-EOS-"
    if len(predictions) == 3 and predictions[1] == "ROOT":
        predictions[1] = "NONE"
    return predictions


def decode(
    sequence_file: Path | str,
    gold_trees: Path | str,
    output_file: Path | str,
) -> Path:
    """Decode predicted labels to one parenthesized tree per line."""
    try:
        from vendor.tree2labels_py3 import utils as tree2labels_utils
    except ImportError as exc:  # pragma: no cover - optional research stack
        raise RuntimeError(
            "Constituency decoding requires NumPy and NLTK; install requirements.txt."
        ) from exc

    gold_lines = Path(gold_trees).read_text(encoding="utf-8").splitlines()
    if not gold_lines:
        raise ValueError(f"Gold tree file is empty: {gold_trees}")
    sentences = []
    predictions = []
    raw_sentences = Path(sequence_file).read_text(encoding="utf-8").split("\n\n")
    for raw_sentence in raw_sentences:
        lines = [line for line in raw_sentence.splitlines() if line]
        if not lines:
            continue
        sentences.append(tree2labels_utils.rebuild_input_sentence(lines))
        predictions.append(_postprocess_labels([line.split("\t")[-1] for line in lines]))
    trees = tree2labels_utils.sequence_to_parenthesis(
        sentences, predictions, join_char="~", split_char="@"
    )
    if gold_lines[0].startswith("( ("):
        trees = [f"( {tree})" for tree in trees]
    if len(trees) != len(gold_lines):
        raise ValueError(
            f"Decoded {len(trees)} trees, but the gold file contains {len(gold_lines)}."
        )
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(trees) + "\n", encoding="utf-8")
    return output


def _backend(model_id: str):
    return canine if model_id.startswith("google/canine-") else subword_models


def train(
    language: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str = "const",
    task: str = "single",
    not_ft_lr: float = 2e-3,
    ft_lr: float = 5e-5,
    epochs: int = 10,
    seed: int = 13,
    *,
    corpus_root: Path | str,
    device: str = "auto",
    encoded_root: Path | str = DEFAULT_ENCODED_ROOT,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
):
    data_dir = encoded_experiment_dir(
        encoding, finetuned, pretrained, language, task, root=encoded_root
    )
    encode(language, task, data_dir, corpus_root=corpus_root)
    util.create_dataset(
        language,
        finetuned,
        pretrained,
        encoding,
        task,
        encoded_root=encoded_root,
        artifact_root=artifact_root,
    )
    return _backend(lm).train(
        language,
        lm,
        finetuned,
        pretrained,
        encoding,
        task=task,
        not_ft_lr=not_ft_lr,
        ft_lr=ft_lr,
        epochs=epochs,
        seed=seed,
        device=device,
        encoded_root=encoded_root,
        artifact_root=artifact_root,
    )


def predict(
    language: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str = "const",
    task: str = "single",
    *,
    device: str = "auto",
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    return _backend(lm).predict(
        language,
        lm,
        finetuned,
        pretrained,
        encoding,
        task=task,
        device=device,
        artifact_root=artifact_root,
    )


def _evalb_binary(language: str, tree2labels_dir: Path | str) -> Path:
    root = require_path(tree2labels_dir, "tree2labels checkout")
    if language.lower() in {"english", "chinese"}:
        binary = root / "EVALB" / "evalb"
    else:
        binary = root / "EVAL_SPRML" / "evalb_spmrl2013.final" / "evalb_spmrl"
    if not binary.is_file():
        raise FileNotFoundError(f"EVALB executable not found: {binary}")
    return binary


def _parse_evalb(output: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().replace(" ", "")
        value = value.strip()
        if key and value:
            results[key] = value
    if "BracketingFMeasure" not in results:
        raise RuntimeError("EVALB output did not contain Bracketing FMeasure.")
    return results


def evaluate(
    language: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str = "const",
    task: str = "single",
    *,
    corpus_root: Path | str,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    tree2labels_dir: Path | str = EXTERNAL_ROOT / "tree2labels",
    scores_path: Path | str | None = None,
) -> dict[str, str]:
    gold = find_constituency_splits(language, corpus_root)["test"]
    experiment = model_dir(
        encoding,
        finetuned,
        pretrained,
        lm,
        language,
        task,
        root=artifact_root,
    )
    sequence_file = experiment / "output" / "test.seq"
    if not sequence_file.is_file():
        raise FileNotFoundError(f"Prediction file not found: {sequence_file}")
    predicted_trees = decode(sequence_file, gold, experiment / "output" / "test.trees")
    completed = subprocess.run(
        [str(_evalb_binary(language, tree2labels_dir)), str(gold), str(predicted_trees)],
        check=True,
        capture_output=True,
        text=True,
    )
    results = _parse_evalb(completed.stdout)
    destination = (
        Path(scores_path)
        if scores_path is not None
        else Path(artifact_root) / "results" / "const_scores.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "language": language,
        "lm": lm,
        "finetuned": finetuned,
        "pretrained": pretrained,
        "encoding": encoding,
        "task": task,
        **results,
    }
    new_file = not destination.exists()
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return results


def evaluate_spans(
    language: str,
    encoding: str,
    finetuned: str,
    pretrained: str,
    *,
    corpus_root: Path | str,
    lms: tuple[str, ...] = (
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "google/canine-c",
        "google/canine-s",
    ),
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    gold = find_constituency_splits(language, corpus_root)["test"]
    output = (
        Path(artifact_root)
        / "plots"
        / "spans"
        / encoding
        / finetuned
        / pretrained
        / f"{language}_spans.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="constituency-predictions-") as temp_dir:
        predicted_root = Path(temp_dir)
        for model_id in lms:
            sequence_file = model_dir(
                encoding,
                finetuned,
                pretrained,
                model_id,
                language,
                root=artifact_root,
            ) / "output" / "test.seq"
            decode(
                sequence_file, gold, predicted_root / model_id.split("/")[-1]
            )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_spans.py"),
                "--gold",
                str(gold),
                "--predicted",
                str(predicted_root),
                "--output",
                str(output),
            ],
            check=True,
        )
    return output
