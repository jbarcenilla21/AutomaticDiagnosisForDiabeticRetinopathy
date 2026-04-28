from sklearn import metrics as sk_metrics


def compute_auc(labels, scores):
    """Compute AUC-ROC score.

    Args:
        labels: array-like of binary ground-truth labels (0 or 1).
        scores: array-like of predicted DR probabilities.

    Returns:
        float AUC value in [0, 1].
    """
    return sk_metrics.roc_auc_score(labels, scores)
