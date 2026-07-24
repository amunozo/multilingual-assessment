import pytest

from scripts.evaluate_dependencies import (
    dependency_head_performance,
    displacement_labeled_performance,
    f1,
    read_conllu,
)
from scripts.evaluate_spans import _validate_lengths
from scripts.plot import relative_error_reduction


def test_dependency_relation_scores_include_predicted_only_relation():
    scores = dependency_head_performance(
        ["2_nsubj", "0_root"], ["2_obj", "0_root"]
    )
    assert scores["nsubj"] == {"p": 0.0, "r": 0.0}
    assert scores["obj"] == {"p": 0.0, "r": 0.0}
    assert scores["root"] == {"p": 1.0, "r": 1.0}


def test_displacement_scores_group_precision_by_predicted_distance():
    scores = displacement_labeled_performance(
        ["1_dep", "1_dep"], ["1_dep", "2_dep"], minimum_frequency=1
    )
    assert scores[1]["r"] == 0.5
    assert scores[2]["p"] == 0.0


def test_read_conllu_ignores_multiword_tokens():
    sentence = (
        "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tca\t_\tAUX\t_\t_\t2\taux\t_\t_\n"
        "2\tn't\t_\tPART\t_\t_\t0\troot\t_\t_\n"
    )
    assert [word.word for word in read_conllu(sentence)] == ["-ROOT-", "ca", "n't"]


def test_analysis_helpers():
    assert f1(0.5, 0.5) == 0.5
    assert relative_error_reduction(90, 80) == 50
    with pytest.raises(ValueError, match="Predicted 1 trees"):
        _validate_lengths(["one"], ["one", "two"])
