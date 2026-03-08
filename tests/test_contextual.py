"""Tests for contextual.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from pet_behavior_clip.contextual import (
    add_context_suffix,
    aggregate_sequence_scores,
    compute_ece_from_labeled_scores,
    estimate_night_probability,
    mix_day_night_scores,
)


class TestContextual(unittest.TestCase):

    def test_estimate_night_probability_dark_vs_bright(self):
        dark = Image.new("RGB", (16, 16), color=(10, 10, 10))
        bright = Image.new("RGB", (16, 16), color=(240, 240, 240))
        probs = estimate_night_probability([dark, bright])
        self.assertEqual(probs.shape, (2,))
        self.assertGreater(probs[0], probs[1])

    def test_add_context_suffix(self):
        pm = {"Active": ["dog moving"]}
        out = add_context_suffix(pm, "captured during daytime")
        self.assertIn("Active", out)
        self.assertIn("captured during daytime", out["Active"][0])

    def test_mix_day_night_scores(self):
        day = pd.DataFrame({"timestamp": [0.0, 1.0], "A": [0.2, 0.8]})
        night = pd.DataFrame({"timestamp": [0.0, 1.0], "A": [0.9, 0.1]})
        p_night = np.array([1.0, 0.0])
        mixed = mix_day_night_scores(day, night, p_night)
        self.assertAlmostEqual(float(mixed.loc[0, "A"]), 0.9)
        self.assertAlmostEqual(float(mixed.loc[1, "A"]), 0.8)

    def test_aggregate_sequence_scores_prob(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "A": [0.0, 1.0, 0.0],
            }
        )
        out = aggregate_sequence_scores(df, mode="prob", window=3)
        self.assertAlmostEqual(float(out.loc[1, "A"]), 1.0 / 3.0, places=6)

    def test_aggregate_sequence_scores_logit_bounds(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "A": [0.01, 0.99, 0.01],
            }
        )
        out = aggregate_sequence_scores(df, mode="logit", window=3)
        self.assertTrue((out["A"] > 0).all())
        self.assertTrue((out["A"] < 1).all())

    def test_compute_ece_from_labeled_scores(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0, 3.0],
                "Active": [0.9, 0.8, 0.2, 0.1],
                "Resting": [0.1, 0.2, 0.8, 0.9],
                "gt_label": ["Active", "Active", "Resting", "Resting"],
            }
        )
        ece = compute_ece_from_labeled_scores(df, label_col="gt_label", n_bins=5)
        self.assertIsNotNone(ece)
        self.assertGreaterEqual(float(ece), 0.0)


if __name__ == "__main__":
    unittest.main()
