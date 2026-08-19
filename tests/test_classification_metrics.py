import pytest

from Network.ClassificationMetrics import binary_classification_metrics


def test_binary_classification_metrics_are_percentage_based():
    metrics = binary_classification_metrics(tp=8, tn=9, fp=1, fn=2)

    assert metrics['accuracy'] == 85.0
    assert metrics['precision'] == pytest.approx(88.8888889)
    assert metrics['recall'] == 80.0
    assert metrics['f1'] == pytest.approx(84.2105263)
    assert metrics['samples'] == 20


def test_binary_classification_metrics_handle_empty_input():
    metrics = binary_classification_metrics(tp=0, tn=0, fp=0, fn=0)

    assert metrics['accuracy'] == 0.0
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['f1'] == 0.0
