"""Subword-model label alignment and probing wrappers."""

from __future__ import annotations

from .modeling import predict_token_classifier, train_token_classifier


def align_labels_with_tokens(labels, word_ids):
    """Keep a word label on its first subtoken and ignore all other tokens."""
    aligned = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != current_word:
            if word_id >= len(labels):
                raise ValueError(f"Tokenizer returned out-of-range word index {word_id}.")
            aligned.append(labels[word_id])
        else:
            aligned.append(-100)
        current_word = word_id
    return aligned


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    tokenized["labels"] = [
        align_labels_with_tokens(labels, tokenized.word_ids(index))
        for index, labels in enumerate(examples["syntax_labels"])
    ]
    return tokenized


def train(treebank, lm, finetuned, pretrained, encoding, **kwargs):
    return train_token_classifier(
        treebank=treebank,
        model_id=lm,
        finetuned=finetuned,
        pretrained=pretrained,
        encoding=encoding,
        align_function=tokenize_and_align_labels,
        **kwargs,
    )


def predict(treebank, lm, finetuned, pretrained, encoding, **kwargs):
    return predict_token_classifier(
        treebank=treebank,
        model_id=lm,
        finetuned=finetuned,
        pretrained=pretrained,
        encoding=encoding,
        align_function=tokenize_and_align_labels,
        **kwargs,
    )
