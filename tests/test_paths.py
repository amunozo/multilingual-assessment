from src.paths import dataset_dir, encoded_experiment_dir, model_dir, model_storage_key


def test_experiment_paths_preserve_model_namespace(tmp_path):
    assert encoded_experiment_dir(
        "const", "not_finetuned", "pretrained", "english", root=tmp_path
    ).relative_to(tmp_path).as_posix() == "const/not_finetuned/pretrained/english/single"
    assert dataset_dir("const", "english", root=tmp_path).relative_to(
        tmp_path
    ).as_posix() == "datasets/const/english/single"
    assert model_dir(
        "const",
        "not_finetuned",
        "pretrained",
        "google/canine-c",
        "english",
        root=tmp_path,
    ).relative_to(tmp_path).as_posix().endswith("google/canine-c/english/single")


def test_local_model_path_cannot_escape_artifact_root(tmp_path):
    local_model = tmp_path / "checkpoints" / "model"
    key = model_storage_key(str(local_model.resolve()))
    assert key.parts[0] == "_local"
    assert not key.is_absolute()
