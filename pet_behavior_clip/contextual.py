"""contextual.py - Context-aware score mixing and sequence aggregation helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def estimate_night_probability(
    frames: Sequence[Image.Image],
    midpoint: float = 0.35,
    sharpness: float = 12.0,
) -> np.ndarray:
    """Estimate per-frame night probability from image brightness.

    Darker frames map to higher night probability.
    """
    if not frames:
        return np.array([], dtype=float)

    means = []
    for img in frames:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        means.append(float(arr.mean()))

    brightness = np.asarray(means, dtype=float)
    logits = sharpness * (brightness - midpoint)
    return 1.0 / (1.0 + np.exp(logits))


def add_context_suffix(
    prompt_map: Dict[str, Sequence[str]],
    suffix: str,
) -> Dict[str, List[str]]:
    """Append a context suffix to each prompt in the map."""
    out: Dict[str, List[str]] = {}
    for label, prompts in prompt_map.items():
        out[label] = [f"{p}, {suffix}" for p in prompts]
    return out


def mix_day_night_scores(
    day_scores: pd.DataFrame,
    night_scores: pd.DataFrame,
    night_probability: Sequence[float],
) -> pd.DataFrame:
    """Mix class probabilities with per-frame night weights.

    output = p_night * night_scores + (1 - p_night) * day_scores
    """
    if len(day_scores) != len(night_scores):
        raise ValueError("day_scores and night_scores must have the same number of rows")
    if len(day_scores) != len(night_probability):
        raise ValueError("night_probability length must match score rows")

    if "timestamp" not in day_scores.columns or "timestamp" not in night_scores.columns:
        raise ValueError("Scores DataFrames must include a 'timestamp' column")

    cols = [c for c in day_scores.columns if c != "timestamp"]
    out = pd.DataFrame({"timestamp": day_scores["timestamp"].to_numpy()})

    w = np.asarray(night_probability, dtype=float).reshape(-1, 1)
    day = day_scores[cols].to_numpy(dtype=float)
    night = night_scores[cols].to_numpy(dtype=float)
    mixed = w * night + (1.0 - w) * day

    for idx, col in enumerate(cols):
        out[col] = mixed[:, idx]
    return out


def aggregate_sequence_scores(
    scores: pd.DataFrame,
    mode: str = "none",
    window: int = 5,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Aggregate frame scores over temporal windows.

    Modes:
    - none: no aggregation
    - prob: rolling mean in probability space
    - logit: inverse-sigmoid -> rolling mean -> sigmoid
    """
    if mode == "none" or scores.empty:
        return scores.copy()

    if "timestamp" not in scores.columns:
        raise ValueError("scores must include a 'timestamp' column")

    cols = [c for c in scores.columns if c != "timestamp"]
    out = scores.copy()

    if mode == "prob":
        out[cols] = scores[cols].rolling(window=window, min_periods=1, center=True).mean()
        return out

    if mode == "logit":
        probs = np.clip(scores[cols].to_numpy(dtype=float), eps, 1.0 - eps)
        logits = np.log(probs / (1.0 - probs))
        logit_df = pd.DataFrame(logits, columns=cols)
        agg_logits = logit_df.rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        out_probs = 1.0 / (1.0 + np.exp(-agg_logits))
        out[cols] = out_probs
        return out

    raise ValueError(f"Unsupported sequence aggregation mode: {mode}")


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
