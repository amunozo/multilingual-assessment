from pathlib import Path

import pytest

from src.const import decode as decode_constituency
from src.const import encode as encode_constituency
from src.const import find_constituency_splits
from src.dep import encode as encode_dependency
from src.dep import find_ud_splits


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_find_ud_splits(tmp_path):
    treebank = tmp_path / "UD_Test-Sample"
    for split in ("train", "dev", "test"):
        _touch(treebank / f"xx_sample-ud-{split}.conllu")
    files = find_ud_splits(tmp_path, "UD_Test-Sample")
    assert files["validation"].name.endswith("dev.conllu")


def test_find_ud_splits_rejects_missing_split(tmp_path):
    treebank = tmp_path / "UD_Test-Sample"
    _touch(treebank / "xx_sample-ud-train.conllu")
    with pytest.raises(FileNotFoundError, match="dev.conllu"):
        find_ud_splits(tmp_path, "UD_Test-Sample")


def test_find_english_constituency_splits(tmp_path):
    for name in ("train.trees", "dev.trees", "test.trees"):
        _touch(tmp_path / name)
    files = find_constituency_splits("english", tmp_path)
    assert files["test"] == tmp_path / "test.trees"


def test_find_spmrl_splits(tmp_path):
    base = tmp_path / "GERMAN_SPMRL" / "gold" / "ptb"
    for split in ("train", "dev", "test"):
        _touch(base / split / f"{split}.German.gold.ptb")
    files = find_constituency_splits("german", tmp_path)
    assert files["validation"].name == "dev.German.gold.ptb"


def test_dependency_encoder_runs_on_tiny_treebank(tmp_path):
    treebank = tmp_path / "ud" / "UD_Test-Sample"
    conllu = (
        "1\tCats\tcat\tNOUN\tNNS\t_\t2\tnsubj\t_\t_\n"
        "2\tsleep\tsleep\tVERB\tVBP\t_\t0\troot\t_\t_\n\n"
    )
    for split in ("train", "dev", "test"):
        path = treebank / f"xx_sample-ud-{split}.conllu"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(conllu, encoding="utf-8")
    outputs = encode_dependency(
        "UD_Test-Sample",
        "single",
        "rel-pos",
        tmp_path / "encoded-dep",
        ud_root=tmp_path / "ud",
    )
    assert "Cats\tNOUN" in outputs["train"].read_text(encoding="utf-8")


def test_constituency_encoder_and_decoder_run_on_tiny_treebank(tmp_path):
    tree = "(S (NP (NNS Cats)) (VP (VBP sleep)))\n"
    for name in ("train.trees", "dev.trees", "test.trees"):
        (tmp_path / name).write_text(tree, encoding="utf-8")
    outputs = encode_constituency(
        "english", "single", tmp_path / "encoded-const", corpus_root=tmp_path
    )
    encoded = outputs["test"].read_text(encoding="utf-8")
    assert "Cats\tNNS" in encoded
    decoded = decode_constituency(
        outputs["test"], tmp_path / "test.trees", tmp_path / "decoded.trees"
    )
    assert len(decoded.read_text(encoding="utf-8").splitlines()) == 1
