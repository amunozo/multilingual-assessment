import pytest

from src.canine import character_labels
from src.subword_models import align_labels_with_tokens


def test_subword_alignment_keeps_only_first_subtoken():
    assert align_labels_with_tokens([3, 7], [None, 0, 0, 1, None]) == [
        -100,
        3,
        -100,
        7,
        -100,
    ]


def test_subword_alignment_rejects_bad_word_index():
    with pytest.raises(ValueError, match="out-of-range"):
        align_labels_with_tokens([1], [0, 1])


def test_canine_alignment_labels_first_character():
    assert character_labels(["Hi", "!"], [4, 9]) == [4, -100, 9]


def test_canine_alignment_rejects_empty_tokens():
    with pytest.raises(ValueError, match="empty tokens"):
        character_labels([""], [1])
