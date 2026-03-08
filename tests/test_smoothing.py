"""Tests for smoothing.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_scores(n: int = 20, labels=("sit", "run", "bark")) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {"timestamp": np.arange(n, dtype=float)}
    for label in labels:
        data[label] = rng.uniform(0, 1, n)
    return pd.DataFrame(data)


class TestSmoothScores(unittest.TestCase):

    def setUp(self):
        from pet_behavior_clip.smoothing import smooth_scores

        self.smooth = smooth_scores

    def test_rolling_mean_shape(self):
        df = _make_scores(20)
        out = self.smooth(df, window=5, method="rolling_mean")
        self.assertEqual(out.shape, df.shape)

    def test_gaussian_shape(self):
        df = _make_scores(20)
        out = self.smooth(df, window=5, method="gaussian", sigma=1.0)
        self.assertEqual(out.shape, df.shape)

    def test_exponential_shape(self):
        df = _make_scores(20)
        out = self.smooth(df, window=5, method="exponential", alpha=0.3)
        self.assertEqual(out.shape, df.shape)

    def test_timestamp_preserved(self):
        df = _make_scores(10)
        out = self.smooth(df, window=3)
        pd.testing.assert_series_equal(df["timestamp"], out["timestamp"])

    def test_invalid_method_raises(self):
        df = _make_scores(10)
        with self.assertRaises(ValueError):
            self.smooth(df, method="invalid_method")

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "sit"])
        out = self.smooth(df)
        self.assertTrue(out.empty)

    def test_rolling_mean_reduces_variance(self):
        """Smoothed signal should have lower variance than raw."""
        df = _make_scores(50)
        out = self.smooth(df, window=7, method="rolling_mean")
        for col in ["sit", "run", "bark"]:
            self.assertLessEqual(out[col].std(), df[col].std() + 1e-9)

    def test_window_1_rolling_mean_unchanged(self):
        """Window size 1 → output equals input for rolling_mean."""
        df = _make_scores(10)
        out = self.smooth(df, window=1, method="rolling_mean")
        for col in ["sit", "run", "bark"]:
            np.testing.assert_allclose(df[col].values, out[col].values, atol=1e-10)


class TestGaussianKernel(unittest.TestCase):

    def test_kernel_sums_to_one(self):
        from pet_behavior_clip.smoothing import _gaussian_kernel

        for window in [3, 5, 7, 11]:
            k = _gaussian_kernel(window, sigma=1.0)
            self.assertAlmostEqual(k.sum(), 1.0, places=10)

    def test_kernel_symmetric(self):
        from pet_behavior_clip.smoothing import _gaussian_kernel

        k = _gaussian_kernel(7, sigma=2.0)
        np.testing.assert_allclose(k, k[::-1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
