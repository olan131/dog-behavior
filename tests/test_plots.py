"""Tests for plots.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_detected(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "timestamp": np.linspace(0, 19, n),
            "sit": rng.uniform(0, 1, n),
            "run": rng.uniform(0, 1, n),
            "bark": rng.uniform(0, 1, n),
            "anomaly_score": rng.uniform(0, 3, n),
        }
    )
    df["is_anomaly"] = df["anomaly_score"] > 2.5
    return df


class TestPlots(unittest.TestCase):

    def test_timeline_returns_figure(self):
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_behavior_timeline

        df = _make_detected()
        fig = plot_behavior_timeline(df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_heatmap_returns_figure(self):
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_anomaly_heatmap

        df = _make_detected()
        fig = plot_anomaly_heatmap(df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_distribution_returns_figure(self):
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_confidence_distribution

        df = _make_detected()
        fig = plot_confidence_distribution(df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_timeline_saves_file(self):
        import os
        import tempfile
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_behavior_timeline

        df = _make_detected()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_behavior_timeline(df, output_path=tmp)
            self.assertTrue(os.path.exists(tmp))
            self.assertGreater(os.path.getsize(tmp), 0)
        finally:
            plt.close("all")
            os.unlink(tmp)

    def test_heatmap_saves_file(self):
        import os
        import tempfile
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_anomaly_heatmap

        df = _make_detected()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_anomaly_heatmap(df, output_path=tmp)
            self.assertTrue(os.path.exists(tmp))
        finally:
            plt.close("all")
            os.unlink(tmp)

    def test_distribution_saves_file(self):
        import os
        import tempfile
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_confidence_distribution

        df = _make_detected()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_confidence_distribution(df, output_path=tmp)
            self.assertTrue(os.path.exists(tmp))
        finally:
            plt.close("all")
            os.unlink(tmp)

    def test_no_anomaly_column(self):
        """Plots should work even without the is_anomaly column."""
        import matplotlib.pyplot as plt
        from pet_behavior_clip.plots import plot_behavior_timeline

        df = _make_detected().drop(columns=["is_anomaly", "anomaly_score"])
        fig = plot_behavior_timeline(df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
