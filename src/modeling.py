"""Shared training and prediction implementation for the probing models."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Callable

from . import util
from .paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ENCODED_ROOT,
    dataset_dir,
    encoded_experiment_dir,
    model_dir,
)


AlignFunction = Callable[[dict[str, object], object], dict[str, object]]


def _select_device(device: str) -> None:
    if device == "auto":
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if device.startswith("cuda:") and device[5:].isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device[5:]
        return
    raise ValueError("device must be 'auto', 'cpu', or 'cuda:N'")


def _stack():
    try:
        import evaluate
        import numpy as np
        from datasets import load_from_disk
        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - optional research stack
        raise RuntimeError(
            "Install the research dependencies with `pip install -r requirements.txt`."
        ) from exc
    return {
        "evaluate": evaluate,
        "np": np,
        "load_from_disk": load_from_disk,
        "AutoConfig": AutoConfig,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForTokenClassification": DataCollatorForTokenClassification,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _training_arguments(
    training_arguments,
    output_dir: Path,
    *,
    learning_rate: float,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
):
    parameters = inspect.signature(training_arguments.__init__).parameters
    kwargs = {
        "per_device_train_batch_size": train_batch_size,
        "gradient_accumulation_steps": 2,
        "per_device_eval_batch_size": eval_batch_size,
        "eval_accumulation_steps": 64,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "save_strategy": "no",
        "report_to": [],
    }
    strategy_name = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_name] = "epoch"
    if device == "cpu":
        if "use_cpu" in parameters:
            kwargs["use_cpu"] = True
        elif "no_cuda" in parameters:
            kwargs["no_cuda"] = True
    return training_arguments(output_dir=str(output_dir), **kwargs)


def _trainer(trainer_class, *, tokenizer, **kwargs):
    parameters = inspect.signature(trainer_class.__init__).parameters
    tokenizer_name = "processing_class" if "processing_class" in parameters else "tokenizer"
    kwargs[tokenizer_name] = tokenizer
    return trainer_class(**kwargs)


def train_token_classifier(
    *,
    treebank: str,
    model_id: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    align_function: AlignFunction,
    task: str = "single",
    not_ft_lr: float = 2e-3,
    ft_lr: float = 5e-5,
    epochs: int = 20,
    train_batch_size: int = 32,
    eval_batch_size: int = 4,
    device: str = "auto",
    seed: int = 13,
    encoded_root: Path | str = DEFAULT_ENCODED_ROOT,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Train one token-classification probe and return its model directory."""
    if finetuned not in {"finetuned", "not_finetuned"}:
        raise ValueError("finetuned must be 'finetuned' or 'not_finetuned'")
    if pretrained not in {"pretrained", "not_pretrained"}:
        raise ValueError("pretrained must be 'pretrained' or 'not_pretrained'")
    _select_device(device)
    stack = _stack()
    stack["set_seed"](seed)

    source_dir = encoded_experiment_dir(
        encoding, finetuned, pretrained, treebank, task, root=encoded_root
    )
    data_files = util.sequence_data_files(source_dir, encoding, treebank)
    class_names = util.return_class_names(data_files)
    labels = class_names["syntax_labels"]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    dataset_path = dataset_dir(encoding, treebank, task, root=artifact_root)
    if not dataset_path.exists():
        util.create_dataset(
            treebank,
            finetuned,
            pretrained,
            encoding,
            task,
            encoded_root=encoded_root,
            artifact_root=artifact_root,
        )
    dataset = stack["load_from_disk"](dataset_path)
    tokenizer = stack["AutoTokenizer"].from_pretrained(model_id)
    tokenized = dataset.map(
        align_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    model_kwargs = {
        "num_labels": len(labels),
        "id2label": id2label,
        "label2id": label2id,
    }
    if pretrained == "pretrained":
        model = stack["AutoModelForTokenClassification"].from_pretrained(
            model_id, **model_kwargs
        )
    else:
        config = stack["AutoConfig"].from_pretrained(model_id, **model_kwargs)
        model = stack["AutoModelForTokenClassification"].from_config(config)

    if finetuned == "not_finetuned":
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False

    output = model_dir(
        encoding,
        finetuned,
        pretrained,
        model_id,
        treebank,
        task,
        root=artifact_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    learning_rate = ft_lr if finetuned == "finetuned" else not_ft_lr
    args = _training_arguments(
        stack["TrainingArguments"],
        output,
        learning_rate=learning_rate,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        device=device,
    )
    metric = stack["evaluate"].load("seqeval")

    def compute_metrics(eval_prediction):
        logits, gold_ids = eval_prediction
        predictions = stack["np"].argmax(logits, axis=-1)
        gold = [[id2label[int(item)] for item in row if item != -100] for row in gold_ids]
        predicted = [
            [id2label[int(pred)] for pred, gold_id in zip(row, gold_row) if gold_id != -100]
            for row, gold_row in zip(predictions, gold_ids)
        ]
        result = metric.compute(predictions=predicted, references=gold)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    collator = stack["DataCollatorForTokenClassification"](tokenizer=tokenizer)
    trainer = _trainer(
        stack["Trainer"],
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    trainer.save_model(output)
    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "treebank": treebank,
            "model_id": model_id,
            "encoding": encoding,
            "finetuned": finetuned,
            "pretrained": pretrained,
            "seed": seed,
        }
    )
    (output / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return output


def predict_token_classifier(
    *,
    treebank: str,
    model_id: str,
    finetuned: str,
    pretrained: str,
    encoding: str,
    align_function: AlignFunction,
    task: str = "single",
    eval_batch_size: int = 8,
    device: str = "auto",
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Predict one test split and write the original three-column format."""
    _select_device(device)
    stack = _stack()
    model_path = model_dir(
        encoding,
        finetuned,
        pretrained,
        model_id,
        treebank,
        task,
        root=artifact_root,
    )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    dataset_path = dataset_dir(encoding, treebank, task, root=artifact_root)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")

    model = stack["AutoModelForTokenClassification"].from_pretrained(model_path)
    tokenizer = stack["AutoTokenizer"].from_pretrained(model_id)
    dataset = stack["load_from_disk"](dataset_path)
    tokenized = dataset.map(
        align_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    output_dir = model_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = inspect.signature(stack["TrainingArguments"].__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "disable_tqdm": True,
        "do_train": False,
        "do_eval": False,
        "do_predict": True,
        "per_device_eval_batch_size": eval_batch_size,
        "report_to": [],
    }
    if device == "cpu":
        if "use_cpu" in parameters:
            kwargs["use_cpu"] = True
        elif "no_cuda" in parameters:
            kwargs["no_cuda"] = True
    args = stack["TrainingArguments"](**kwargs)
    collator = stack["DataCollatorForTokenClassification"](tokenizer=tokenizer)
    trainer = _trainer(
        stack["Trainer"],
        tokenizer=tokenizer,
        model=model,
        args=args,
        data_collator=collator,
    )
    result = trainer.predict(tokenized["test"])
    prediction_ids = stack["np"].argmax(result.predictions, axis=-1)
    predicted_labels = [
        [model.config.id2label[int(pred)] for pred, gold in zip(row, gold_row) if gold != -100]
        for row, gold_row in zip(prediction_ids, result.label_ids)
    ]

    pos_feature = dataset["train"].features["pos_tags"].feature
    output_lines: list[str] = []
    for sentence_index, (tokens, pos_ids, labels) in enumerate(
        zip(dataset["test"]["tokens"], dataset["test"]["pos_tags"], predicted_labels),
        start=1,
    ):
        if len(tokens) != len(labels):
            raise ValueError(
                f"Sentence {sentence_index} has {len(tokens)} words but {len(labels)} predictions."
            )
        for token, pos_id, label in zip(tokens, pos_ids, labels):
            output_lines.append(f"{token}\t{pos_feature.int2str(pos_id)}\t{label}")
        output_lines.append("")

    output_path = output_dir / "test.seq"
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_path
