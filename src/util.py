"""Utilities for converting sequence-label files to Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from .paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ENCODED_ROOT,
    dataset_dir,
    encoded_experiment_dir,
)


SPLITS = ("train", "validation", "test")


def load_conllu(filename: Path | str):
    """Load a CoNLL-U file, importing the optional parser only when needed."""
    try:
        import conllu
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise RuntimeError("Install the research dependencies with `pip install -r requirements.txt`.") from exc

    with Path(filename).open(encoding="utf-8") as handle:
        return conllu.parse(handle.read())


def _sentences(path: Path | str) -> Iterable[list[tuple[str, str, str]]]:
    """Yield validated token/POS/label triples from a sequence-label file."""
    sentence: list[tuple[str, str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"Expected three tab-separated fields in {path}:{line_number}; "
                    f"found {len(fields)}."
                )
            sentence.append((fields[0], fields[1], fields[2]))
    if sentence:
        yield sentence


def return_class_names(data_files: dict[str, Path | str]) -> dict[str, list[str]]:
    """Return deterministic POS and syntax-label vocabularies.

    Syntax labels are learned from train and validation only. Test-only labels
    are mapped to ``UNK`` when records are created.
    """
    missing = set(SPLITS) - set(data_files)
    if missing:
        raise ValueError(f"Missing dataset splits: {', '.join(sorted(missing))}")

    pos_tags: set[str] = set()
    syntax_labels: set[str] = set()
    for split in SPLITS:
        for sentence in _sentences(data_files[split]):
            for _, pos_tag, syntax_label in sentence:
                pos_tags.add(pos_tag)
                if split != "test":
                    syntax_labels.add(syntax_label)

    labels = sorted(syntax_labels - {"UNK"})
    labels.append("UNK")
    return {"pos_tags": sorted(pos_tags), "syntax_labels": labels}


def sequence_records(
    path: Path | str,
    *,
    split: str,
    known_labels: set[str],
) -> list[dict[str, object]]:
    """Convert one sequence-label split to JSON-compatible records."""
    records: list[dict[str, object]] = []
    for index, sentence in enumerate(_sentences(path), start=1):
        tokens, pos_tags, labels = zip(*sentence)
        if split == "test":
            labels = tuple(label if label in known_labels else "UNK" for label in labels)
        records.append(
            {
                "id": index,
                "tokens": list(tokens),
                "pos_tags": list(pos_tags),
                "syntax_labels": list(labels),
            }
        )
    return records


def create_json_files(
    data_files: dict[str, Path | str], output_dir: Path | str
) -> dict[str, str]:
    """Write the three sequence-label splits as JSON files."""
    class_names = return_class_names(data_files)
    known_labels = set(class_names["syntax_labels"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_files: dict[str, str] = {}
    for split in SPLITS:
        destination = output / f"{split}.json"
        records = sequence_records(
            data_files[split], split=split, known_labels=known_labels
        )
        destination.write_text(
            json.dumps({"data": records}, ensure_ascii=False), encoding="utf-8"
        )
        json_files[split] = str(destination)
    return json_files


def sequence_data_files(data_dir: Path | str, encoding: str, treebank: str) -> dict[str, Path]:
    root = Path(data_dir)
    if encoding == "const":
        names = {
            "train": f"{treebank}-train.seq_lu",
            "validation": f"{treebank}-dev.seq_lu",
            "test": f"{treebank}-test.seq_lu",
        }
    else:
        names = {"train": "train", "validation": "dev", "test": "test"}
    files = {split: root / name for split, name in names.items()}
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing encoded split(s): " + ", ".join(absent))
    return files


def create_dataset(
    treebank: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    task: str = "single",
    *,
    encoded_root: Path | str = DEFAULT_ENCODED_ROOT,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
):
    """Create and persist an untokenized Hugging Face dataset."""
    try:
        from datasets import ClassLabel, Features, Sequence, Value, load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise RuntimeError("Install the research dependencies with `pip install -r requirements.txt`.") from exc

    data_dir = encoded_experiment_dir(
        encoding,
        finetuned,
        pretrained,
        treebank,
        task,
        root=encoded_root,
    )
    data_files = sequence_data_files(data_dir, encoding, treebank)
    class_names = return_class_names(data_files)

    with TemporaryDirectory(prefix="multilingual-assessment-json-") as temp_dir:
        json_files = create_json_files(data_files, temp_dir)
        dataset = load_dataset(
            "json",
            data_files=json_files,
            field="data",
            features=Features(
                {
                    "id": Value("int32"),
                    "tokens": Sequence(Value("string")),
                    "pos_tags": Sequence(ClassLabel(names=class_names["pos_tags"])),
                    "syntax_labels": Sequence(
                        ClassLabel(names=class_names["syntax_labels"])
                    ),
                }
            ),
        )

    output = dataset_dir(encoding, treebank, task, root=artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output)
    return dataset
