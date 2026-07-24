"""CANINE character-label alignment and probing wrappers."""

from __future__ import annotations

from .modeling import predict_token_classifier, train_token_classifier


def character_labels(tokens, labels):
    """Assign each word's label to its first character."""
    if len(tokens) != len(labels):
        raise ValueError("Each word must have exactly one syntax label.")
    aligned = []
    for token, label in zip(tokens, labels):
        if not token:
            raise ValueError("CANINE alignment does not support empty tokens.")
        aligned.append(label)
        aligned.extend([-100] * (len(token) - 1))
    return aligned


def tokenize(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    word_labels = [
        character_labels(tokens, tags)
        for tokens, tags in zip(examples["tokens"], examples["syntax_labels"])
    ]
    labels = []
    for index, (input_ids, expected) in enumerate(
        zip(tokenized["input_ids"], word_labels)
    ):
        special_mask = tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        if special_mask.count(0) != len(expected):
            raise ValueError(
                "CANINE tokenization produced a character/label length mismatch "
                f"for batch item {index}: {special_mask.count(0)} character tokens "
                f"versus {len(expected)} labels."
            )
        expected_iter = iter(expected)
        labels.append(
            [-100 if is_special else next(expected_iter) for is_special in special_mask]
        )
    tokenized["labels"] = labels
    return tokenized


def train(treebank, lm, finetuned, pretrained, encoding, **kwargs):
    return train_token_classifier(
        treebank=treebank,
        model_id=lm,
        finetuned=finetuned,
        pretrained=pretrained,
        encoding=encoding,
        align_function=tokenize,
        **kwargs,
    )


def predict(treebank, lm, finetuned, pretrained, encoding, **kwargs):
    return predict_token_classifier(
        treebank=treebank,
        model_id=lm,
        finetuned=finetuned,
        pretrained=pretrained,
        encoding=encoding,
        align_function=tokenize,
        **kwargs,
    )
