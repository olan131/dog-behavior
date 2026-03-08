"""smoothing.py – Temporal smoothing of per-frame classification scores.

Smoothing reduces the impact of noisy single-frame predictions and produces a
more stable signal for downstream anomaly detection.

Supported methods
-----------------
``rolling_mean``
    Simple sliding-window average.
``gaussian``
    Gaussian-weighted kernel convolution.
``exponential``
    Exponential weighted moving average.
"""

from __future__ import annotations

import logging
from typing import Literal, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SmoothingMethod = Literal["rolling_mean", "gaussian", "exponential"]


def smooth_scores(
    scores: pd.DataFrame,
    window: int = 5,
    method: SmoothingMethod = "rolling_mean",
    sigma: float = 1.0,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """Apply temporal smoothing to a behaviour-score DataFrame.

    Parameters
    ----------
    scores:
        DataFrame produced by :meth:`~pet_behavior_clip.clip_zeroshot.SigLIPClassifier.classify_frames`.
        Must contain a ``timestamp`` column; all other columns are treated as
        numeric score columns.
    window:
        Window size in frames used by ``rolling_mean`` and ``gaussian`` methods.
    method:
        Smoothing algorithm.  One of ``"rolling_mean"``, ``"gaussian"``, or
        ``"exponential"``.
    sigma:
        Standard deviation for the Gaussian kernel (used only when
        ``method="gaussian"``).
    alpha:
        Smoothing factor for ``"exponential"`` EWM (closer to 1 → less smooth).

    Returns
    -------
    pandas.DataFrame
        Same shape as *scores* with smoothed numeric columns.
        The ``timestamp`` column is preserved unmodified.
    """
    if scores.empty:
        return scores.copy()

    score_cols = [c for c in scores.columns if c != "timestamp"]
    result = scores.copy()

    if method == "rolling_mean":
        result[score_cols] = (
            scores[score_cols]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
    elif method == "gaussian":
        kernel = _gaussian_kernel(window, sigma)
        for col in score_cols:
            result[col] = np.convolve(
                scores[col].to_numpy(), kernel, mode="same"
            )
    elif method == "exponential":
        result[score_cols] = scores[score_cols].ewm(alpha=alpha).mean()
    else:
        raise ValueError(
            f"Unknown smoothing method '{method}'. "
            "Choose from: rolling_mean, gaussian, exponential."
        )

    logger.debug("Applied %s smoothing (window=%d) to %d frames.", method, window, len(scores))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel(window: int, sigma: float) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel of the given *window* size."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()
