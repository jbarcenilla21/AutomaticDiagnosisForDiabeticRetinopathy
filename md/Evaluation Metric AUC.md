# Evaluation Metric: AUC

## Why AUC?

We evaluate our binary classifier using the **Area Under the ROC Curve (AUC)**.

AUC has several advantages for this problem:

- **Threshold-independent** — it operates on the soft (probability) outputs of the classifier, so no decision threshold needs to be fixed.
- **Robust to class imbalance** — it evaluates the *ranking* of scores relative to labels, not their absolute values.
- **Interpretable range** — AUC ∈ [0, 1], where:
  - `1.0` → perfect classifier
  - `0.5` → random baseline (equivalent to guessing 0 or 1 with equal probability)
  - `< 0.5` → usually indicates a bug (e.g. inverted outputs)

## How the ROC Curve is Built

By sweeping the decision threshold from 0 to 1, each threshold yields a pair of rates:

| Axis | Metric | Definition |
|------|--------|------------|
| X | **FPR** (False Positive Rate) | False detections / total negatives |
| Y | **TPR** (True Positive Rate) | Correct detections / total positives |

- **Low threshold** → system tends to predict 1 → high TPR but also high FPR.
- **High threshold** → system tends to predict 0 → low FPR but also low TPR.

The ROC curve traces all these (FPR, TPR) pairs, and the **AUC is the integral under that curve**.

## Usage in scikit-learn

```python
from sklearn import metrics

auc = metrics.roc_auc_score(labels, scores)
```

- `labels` — ground-truth binary labels (0: No DR, 1: DR)
- `scores` — soft output of the classifier (e.g. predicted probability of DR)
