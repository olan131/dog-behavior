"""anomaly.py – Anomaly detection on per-frame behaviour scores.

An "anomaly" is a frame (or short temporal window) where the observed
behaviour-score distribution deviates significantly from the baseline.

Two detection strategies are provided:

``zscore``
    Flag frames whose z-score for *any* label exceeds a threshold.
``iqr``
    Flag frames that fall outside ``[Q1 - k·IQR, Q3 + k·IQR]`` for any label.

The detector returns an augmented DataFrame that includes the original scores
plus ``anomaly_score`` (a scalar per frame) and ``is_anomaly`` (bool flag).
"""

from __future__ import annotations

import logging
from typing import Literal, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DetectionMethod = Literal["zscore", "iqr"]


class AnomalyDetector:
    """Detect anomalous frames in a smoothed behaviour-score DataFrame.

    Parameters
    ----------
    method:
        Detection algorithm – ``"zscore"`` (default) or ``"iqr"``.
    threshold:
        * For ``zscore``: z-score magnitude above which a frame is flagged.
          Common values: 2.0 – 3.0.
        * For ``iqr``: the multiplier *k* such that values outside
          ``[Q1 - k·IQR, Q3 + k·IQR]`` are flagged.  Typically 1.5.
    score_cols:
        Explicit list of columns to analyse.  When *None* every column except
        ``timestamp``, ``anomaly_score``, and ``is_anomaly`` is used.
    """

    def __init__(
        self,
        method: DetectionMethod = "zscore",
        threshold: float = 2.5,
        score_cols: List[str] | None = None,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.score_cols = score_cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Annotate *scores* with anomaly information.

        Parameters
        ----------
        scores:
            Output of :func:`~pet_behavior_clip.smoothing.smooth_scores` (or
            directly from :class:`~pet_behavior_clip.clip_zeroshot.SigLIPClassifier`).

        Returns
        -------
        pandas.DataFrame
            Original DataFrame plus two additional columns:

            ``anomaly_score``
                Maximum absolute deviation (z-score or IQR units) across all
                label columns for that frame.
            ``is_anomaly``
                ``True`` when *anomaly_score* exceeds *threshold*.
        """
        if scores.empty:
            result = scores.copy()
            result["anomaly_score"] = pd.Series(dtype=float)
            result["is_anomaly"] = pd.Series(dtype=bool)
            return result

        cols = self._resolve_cols(scores)
        data = scores[cols].to_numpy(dtype=float)

        if self.method == "zscore":
            deviations = self._zscore_deviations(data)
        elif self.method == "iqr":
            deviations = self._iqr_deviations(data)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'zscore' or 'iqr'.")

        anomaly_score = deviations.max(axis=1)
        is_anomaly = anomaly_score > self.threshold

        result = scores.copy()
        result["anomaly_score"] = anomaly_score
        result["is_anomaly"] = is_anomaly

        n_anomalies = int(is_anomaly.sum())
        logger.info(
            "Anomaly detection (%s, threshold=%.2f): %d/%d frames flagged.",
            self.method,
            self.threshold,
            n_anomalies,
            len(scores),
        )
        return result

    def summary(self, detected: pd.DataFrame) -> dict:
        """Return a plain-dict summary of detection results."""
        if "is_anomaly" not in detected.columns:
            raise ValueError("Run detect() first.")
        total = len(detected)
        n_anomaly = int(detected["is_anomaly"].sum())
        duration = (
            float(detected["timestamp"].iloc[-1]) if "timestamp" in detected.columns else None
        )
        worst_ts = None
        if n_anomaly > 0 and "timestamp" in detected.columns:
            worst_ts = float(
                detected.loc[detected["anomaly_score"].idxmax(), "timestamp"]
            )
        return {
            "total_frames": total,
            "anomaly_frames": n_anomaly,
            "anomaly_ratio": round(n_anomaly / total, 4) if total > 0 else 0.0,
            "duration_seconds": duration,
            "worst_anomaly_timestamp": worst_ts,
            "method": self.method,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, scores: pd.DataFrame) -> List[str]:
        if self.score_cols:
            return self.score_cols
        exclude = {"timestamp", "anomaly_score", "is_anomaly"}
        return [c for c in scores.columns if c not in exclude]

    @staticmethod
    def _zscore_deviations(data: np.ndarray) -> np.ndarray:
        """Return per-cell absolute z-scores.  Shape: (N, L)."""
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # avoid division by zero
        return np.abs((data - mean) / std)

    def _iqr_deviations(self, data: np.ndarray) -> np.ndarray:
        """Return per-cell IQR-unit deviations.  Shape: (N, L).

        NOTE: 使用 self.threshold 作為 IQR 倍率 k，
        邊界為 [Q1 - k·IQR, Q3 + k·IQR]。
        """
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        iqr = np.where(iqr == 0, 1.0, iqr)  # avoid division by zero
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        below = np.maximum(0, lower - data) / iqr
        above = np.maximum(0, data - upper) / iqr
        return np.maximum(below, above)
