def binary_classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Return percentage-based binary metrics from confusion-matrix counts.

    Label ``1`` is treated as the positive class. Zero-denominator metrics are
    reported as ``0.0`` so evaluation also works on single-class subsets.
    """

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        'accuracy': accuracy * 100.0,
        'precision': precision * 100.0,
        'recall': recall * 100.0,
        'f1': f1 * 100.0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'samples': int(total),
    }
