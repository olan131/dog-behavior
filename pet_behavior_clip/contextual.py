"""contextual.py - Context-level evaluation helpers for local inference."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
def compute_ece_from_labeled_scores(
    scores: pd.DataFrame,
    label_col: str = "gt_label",
    n_bins: int = 20,
) -> Optional[float]:
    """Compute ECE when labeled ground truth is available.

    Returns ``None`` when labels are unavailable.
    """
    if label_col not in scores.columns:
        return None

    exclude = {"timestamp", "anomaly_score", "is_anomaly", label_col}
    class_cols = [c for c in scores.columns if c not in exclude]
    if not class_cols:
        return None

    probs = scores[class_cols].to_numpy(dtype=float)
    conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred_labels = np.array(class_cols)[pred_idx]
    gt = scores[label_col].astype(str).to_numpy()
    correct = (pred_labels == gt).astype(float)

    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc - avg_conf)

    return float(ece)
