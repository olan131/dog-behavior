"""Tests for anomaly.py (AnomalyDetector)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_scores(n: int = 30, labels=("sit", "run")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"timestamp": np.linspace(0, n - 1, n)}
    for label in labels:
        data[label] = rng.uniform(0.3, 0.7, n)
    return pd.DataFrame(data)


def _inject_spike(df: pd.DataFrame, frame_idx: int, label: str, value: float) -> pd.DataFrame:
    df = df.copy()
    df.loc[frame_idx, label] = value
    return df


class TestAnomalyDetectorZScore(unittest.TestCase):

    def setUp(self):
        from pet_behavior_clip.anomaly import AnomalyDetector

        self.Detector = AnomalyDetector

    def test_output_columns(self):
        det = self.Detector(method="zscore", threshold=2.5)
        df = _make_scores()
        result = det.detect(df)
        self.assertIn("anomaly_score", result.columns)
        self.assertIn("is_anomaly", result.columns)

    def test_output_shape(self):
        det = self.Detector()
        df = _make_scores(25)
        result = det.detect(df)
        self.assertEqual(len(result), 25)

    def test_spike_detected(self):
        """A single extreme spike should be flagged."""
        det = self.Detector(method="zscore", threshold=2.0)
        df = _make_scores(50)
        df = _inject_spike(df, 25, "sit", 5.0)  # very high value
        result = det.detect(df)
        self.assertTrue(result.loc[25, "is_anomaly"])

    def test_normal_data_not_flagged(self):
        """Uniform data should produce no anomalies."""
        det = self.Detector(method="zscore", threshold=3.0)
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            {
                "timestamp": np.arange(100, dtype=float),
                "sit": rng.normal(0.5, 0.01, 100),
            }
        )
        result = det.detect(df)
        self.assertEqual(result["is_anomaly"].sum(), 0)

    def test_empty_dataframe(self):
        det = self.Detector()
        df = pd.DataFrame(columns=["timestamp", "sit"])
        result = det.detect(df)
        self.assertTrue(result.empty)

    def test_summary_keys(self):
        det = self.Detector()
        df = _make_scores(20)
        result = det.detect(df)
        summary = det.summary(result)
        for key in ("total_frames", "anomaly_frames", "anomaly_ratio", "method"):
            self.assertIn(key, summary)

    def test_summary_total_frames(self):
        det = self.Detector()
        df = _make_scores(15)
        result = det.detect(df)
        summary = det.summary(result)
        self.assertEqual(summary["total_frames"], 15)


class TestAnomalyDetectorIQR(unittest.TestCase):

    def setUp(self):
        from pet_behavior_clip.anomaly import AnomalyDetector

        self.Detector = AnomalyDetector

    def test_spike_detected_iqr(self):
        det = self.Detector(method="iqr", threshold=1.5)
        df = _make_scores(50)
        df = _inject_spike(df, 10, "run", 10.0)
        result = det.detect(df)
        self.assertTrue(result.loc[10, "is_anomaly"])

    def test_invalid_method(self):
        from pet_behavior_clip.anomaly import AnomalyDetector

        det = AnomalyDetector(method="invalid")  # type: ignore[arg-type]
        df = _make_scores(5)
        with self.assertRaises(ValueError):
            det.detect(df)

    def test_anomaly_score_non_negative(self):
        det = self.Detector(method="iqr", threshold=1.5)
        df = _make_scores(30)
        result = det.detect(df)
        self.assertTrue((result["anomaly_score"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
