"""Tests for local report generation."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pet_behavior_clip.report import generate_report


def _make_detected(n: int = 20, n_anomalies: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "timestamp": np.linspace(0, 19, n),
            "sit": rng.uniform(0, 1, n),
            "run": rng.uniform(0, 1, n),
            "bark": rng.uniform(0, 1, n),
            "anomaly_score": rng.uniform(0, 3, n),
        }
    )
    flags = np.zeros(n, dtype=bool)
    flags[:n_anomalies] = True
    rng.shuffle(flags)
    df["is_anomaly"] = flags
    return df


class TestGenerateReport(unittest.TestCase):

    def setUp(self):
        self.labels = ["sit", "run", "bark"]

    def test_returns_string(self):
        report = generate_report(_make_detected(), self.labels)
        self.assertIsInstance(report, str)
        self.assertIn("Pet Behavior Analysis Report", report)

    def test_contains_label_stats(self):
        report = generate_report(_make_detected(), self.labels)
        for label in self.labels:
            self.assertIn(label, report)

    def test_report_saved_to_file(self):
        df = _make_detected()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp = f.name
        os.unlink(tmp)
        try:
            generate_report(df, self.labels, output_path=tmp)
            self.assertTrue(os.path.exists(tmp))
            with open(tmp, encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("Pet Behavior Analysis Report", content)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
