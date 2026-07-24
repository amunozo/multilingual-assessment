"""Dependency-parsing probe orchestration."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

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


SPLIT_SUFFIXES = {
    "train": "train.conllu",
    "validation": "dev.conllu",
    "test": "test.conllu",
}


def find_ud_splits(ud_root: Path | str, treebank: str) -> dict[str, Path]:
    """Locate exactly one train, development, and test file for a UD treebank."""
    treebank_dir = require_path(Path(ud_root) / treebank, "UD treebank directory")
    result: dict[str, Path] = {}
    for split, suffix in SPLIT_SUFFIXES.items():
        matches = sorted(treebank_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *{suffix} file in {treebank_dir}; found {len(matches)}."
            )
        result[split] = matches[0]
    return result


def _labeler(dep2label_dir: Path | str):
    dependency_root = require_path(dep2label_dir, "dep2label checkout")
    if not (dependency_root / "dep2label" / "labeling.py").is_file():
        raise FileNotFoundError(
            f"dep2label package not found under {dependency_root}; run `git submodule update --init --recursive`."
        )
    root_text = str(dependency_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from dep2label.labeling import Labeler

    return Labeler


def encode(
    treebank: str,
    task: str,
    encoding: str,
    output_dir: Path | str,
    *,
    ud_root: Path | str,
    dep2label_dir: Path | str = EXTERNAL_ROOT / "dep2label",
) -> dict[str, Path]:
    """Encode the three UD splits as sequence-label files."""
    if task != "single":
        raise NotImplementedError("Only the single-task setup is supported by this pipeline.")
    labeler_class = _labeler(dep2label_dir)
    source_files = find_ud_splits(ud_root, treebank)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    names = {"train": "train", "validation": "dev", "test": "test"}
    destinations: dict[str, Path] = {}
    for split, source in source_files.items():
        destination = destination_root / names[split]
        labeler_class().encode(str(source), str(destination), encoding, None)
        destinations[split] = destination
    return destinations


def _single_task_text(path: Path | str, task: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if task == "single":
        return text
    if task != "multi":
        raise ValueError("task must be 'single' or 'multi'")
    converted: list[str] = []
    for line in text.splitlines():
        if not line:
            converted.append("")
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"Malformed predicted sequence line: {line!r}")
        converted.append("\t".join(fields[:2]) + "\t" + "{}".join(fields[2:]))
    return "\n".join(converted) + "\n"


def decode(
    file: Path | str,
    encoding: str,
    task: str,
    output_file: Path | str,
    original_conllu: Path | str,
    *,
    dep2label_dir: Path | str = EXTERNAL_ROOT / "dep2label",
) -> Path:
    """Decode predicted sequence labels to CoNLL-U."""
    labeler_class = _labeler(dep2label_dir)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".seq", delete=True
    ) as temporary:
        temporary.write(_single_task_text(file, task))
        temporary.flush()
        labeler_class().decode(
            temporary.name, str(output), encoding, str(original_conllu)
        )
    return output


def _backend(model_id: str):
    return canine if model_id.startswith("google/canine-") else subword_models


def train(
    treebank: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    task: str = "single",
    not_ft_lr: float = 2e-3,
    ft_lr: float = 5e-5,
    epochs: int = 20,
    seed: int = 13,
    *,
    ud_root: Path | str,
    device: str = "auto",
    encoded_root: Path | str = DEFAULT_ENCODED_ROOT,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
):
    """Encode data, build a dataset, and train one dependency probe."""
    data_dir = encoded_experiment_dir(
        encoding, finetuned, pretrained, treebank, task, root=encoded_root
    )
    encode(
        treebank,
        task,
        encoding,
        data_dir,
        ud_root=ud_root,
    )
    util.create_dataset(
        treebank,
        finetuned,
        pretrained,
        encoding,
        task,
        encoded_root=encoded_root,
        artifact_root=artifact_root,
    )
    return _backend(lm).train(
        treebank,
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
    treebank: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    task: str = "single",
    *,
    device: str = "auto",
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Run one dependency probe on its prepared test split."""
    return _backend(lm).predict(
        treebank,
        lm,
        finetuned,
        pretrained,
        encoding,
        task=task,
        device=device,
        artifact_root=artifact_root,
    )


def _append_scores(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def evaluate(
    treebank: str,
    lm: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    task: str = "single",
    *,
    ud_root: Path | str,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    scores_path: Path | str | None = None,
) -> tuple[float, float]:
    """Decode and evaluate one dependency prediction with CoNLL 2018 metrics."""
    from . import conll18_ud_eval as ud_eval

    gold = find_ud_splits(ud_root, treebank)["test"]
    experiment = model_dir(
        encoding,
        finetuned,
        pretrained,
        lm,
        treebank,
        task,
        root=artifact_root,
    )
    predicted_seq = experiment / "output" / "test.seq"
    if not predicted_seq.is_file():
        raise FileNotFoundError(f"Prediction file not found: {predicted_seq}")
    predicted_conllu = experiment / "output" / "test.conllu"
    decode(predicted_seq, encoding, task, predicted_conllu, gold)

    metrics = ud_eval.evaluate(
        ud_eval.load_conllu_file(str(gold)),
        ud_eval.load_conllu_file(str(predicted_conllu)),
    )
    values = {name: metrics[name].f1 for name in ("UAS", "LAS", "CLAS", "MLAS", "BLEX")}
    destination = (
        Path(scores_path)
        if scores_path is not None
        else Path(artifact_root) / "results" / "dep_scores.csv"
    )
    _append_scores(
        destination,
        {
            "Treebank": treebank,
            "Encoding": encoding,
            "Language Model": lm,
            "Finetuned": finetuned,
            "Pretrained": pretrained,
            **{name: round(value * 100, 2) for name, value in values.items()},
        },
    )
    return values["UAS"], values["LAS"]


def evaluate_displacement(
    treebank: str,
    encoding: str,
    *,
    ud_root: Path | str,
    finetuned: str = "not_finetuned",
    pretrained: str = "pretrained",
    lms: tuple[str, ...] = (
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "google/canine-c",
        "google/canine-s",
    ),
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Create the dependency-displacement plot used in the paper analysis."""
    import shutil
    import subprocess

    gold = find_ud_splits(ud_root, treebank)["test"]
    output = (
        Path(artifact_root)
        / "plots"
        / "displacements"
        / encoding
        / finetuned
        / pretrained
        / f"{treebank}_displacements.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dependency-predictions-") as temp_dir:
        predicted_root = Path(temp_dir)
        for model_id in lms:
            source = model_dir(
                encoding,
                finetuned,
                pretrained,
                model_id,
                treebank,
                root=artifact_root,
            ) / "output" / "test.conllu"
            if not source.is_file():
                raise FileNotFoundError(f"Decoded prediction not found: {source}")
            shutil.copy2(source, predicted_root / model_id.split("/")[-1])
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_dependencies.py"),
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
