"""Project paths shared by data preparation, training, and evaluation."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_ENCODED_ROOT = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("MULTILINGUAL_ASSESSMENT_OUTPUT_DIR", PROJECT_ROOT / "artifacts")
).expanduser()


def encoded_experiment_dir(
    encoding: str,
    finetuned: str,
    pretrained: str,
    treebank: str,
    task: str = "single",
    *,
    root: Path | str = DEFAULT_ENCODED_ROOT,
) -> Path:
    return Path(root) / encoding / finetuned / pretrained / treebank / task


def dataset_dir(
    encoding: str,
    treebank: str,
    task: str = "single",
    *,
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    return Path(root) / "datasets" / encoding / treebank / task


def model_dir(
    encoding: str,
    finetuned: str,
    pretrained: str,
    model_id: str,
    treebank: str,
    task: str = "single",
    *,
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    return (
        Path(root)
        / "models"
        / encoding
        / finetuned
        / pretrained
        / model_storage_key(model_id)
        / treebank
        / task
    )


def model_storage_key(model_id: str) -> Path:
    """Map a Hub ID or local model path to a safe, stable artifact path."""
    candidate = Path(model_id).expanduser()
    if candidate.is_absolute() or model_id.startswith(".") or candidate.exists():
        resolved = candidate.resolve()
        digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:10]
        return Path("_local") / f"{resolved.name}-{digest}"
    return Path(model_id)


def require_path(path: Path | str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved
