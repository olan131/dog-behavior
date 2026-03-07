"""Tests for report_llm.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


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
        from pet_behavior_clip.report_llm import generate_report

        self.generate = generate_report
        self.labels = ["sit", "run", "bark"]

    def test_template_returns_string(self):
        df = _make_detected()
        report = self.generate(df, self.labels, mode="template")
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 50)

    def test_template_contains_title(self):
        df = _make_detected()
        report = self.generate(df, self.labels, mode="template")
        self.assertIn("寵物行為分析報告", report)

    def test_template_contains_label_stats(self):
        df = _make_detected()
        report = self.generate(df, self.labels, mode="template")
        for label in self.labels:
            self.assertIn(label, report)

    def test_report_saved_to_file(self):
        import os
        import tempfile

        df = _make_detected()
        with tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, mode="w", encoding="utf-8"
        ) as f:
            tmp = f.name
        os.unlink(tmp)  # remove so generate_report creates it fresh
        try:
            self.generate(df, self.labels, mode="template", output_path=tmp)
            self.assertTrue(os.path.exists(tmp))
            content = open(tmp, encoding="utf-8").read()
            self.assertIn("寵物行為分析報告", content)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_no_anomalies_low_risk(self):
        """Zero anomalies → low-risk report language."""
        df = _make_detected(n=20, n_anomalies=0)
        report = self.generate(df, self.labels, mode="template")
        self.assertIn("低風險", report)

    def test_high_anomaly_ratio_high_risk(self):
        """All frames anomalous → high-risk report."""
        df = _make_detected(n=20, n_anomalies=20)
        report = self.generate(df, self.labels, mode="template")
        self.assertIn("高風險", report)

    def test_llm_fallback_without_api_key(self):
        """When OPENAI_API_KEY is absent, mode=llm should fall back to template."""
        import os
        from unittest.mock import patch

        df = _make_detected()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the key is not set
            os.environ.pop("OPENAI_API_KEY", None)
            report = self.generate(df, self.labels, mode="llm")
        self.assertIn("寵物行為分析報告", report)


if __name__ == "__main__":
    unittest.main()
