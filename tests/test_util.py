from pathlib import Path

import pytest

from src import util


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_class_names_are_sorted_and_ignore_test_only_labels(tmp_path):
    files = {
        "train": _write(tmp_path / "train", "a\tZ\tB\n\nb\tA\tA\n"),
        "validation": _write(tmp_path / "dev", "c\tC\tC\n"),
        "test": _write(tmp_path / "test", "d\tD\tTEST_ONLY\n"),
    }

    assert util.return_class_names(files) == {
        "pos_tags": ["A", "C", "D", "Z"],
        "syntax_labels": ["A", "B", "C", "UNK"],
    }


def test_test_only_label_is_mapped_to_unknown(tmp_path):
    path = _write(tmp_path / "test", "word\tNOUN\tNEW\n")
    records = util.sequence_records(path, split="test", known_labels={"KNOWN", "UNK"})
    assert records[0]["syntax_labels"] == ["UNK"]


def test_sequence_reader_reports_malformed_lines(tmp_path):
    path = _write(tmp_path / "broken", "only\ttwo\n")
    with pytest.raises(ValueError, match="three tab-separated fields"):
        list(util._sentences(path))


def test_sequence_data_files_uses_constituency_names(tmp_path):
    for suffix in ("train", "dev", "test"):
        _write(tmp_path / f"english-{suffix}.seq_lu", "x\tX\tX\n")
    files = util.sequence_data_files(tmp_path, "const", "english")
    assert files["validation"].name == "english-dev.seq_lu"
