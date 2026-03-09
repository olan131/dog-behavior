"""Tests for contextual.py."""

from __future__ import annotations

import unittest

import pandas as pd

from pet_behavior_clip.contextual import (
    aggregate_sequence_scores,
    compute_ece_from_labeled_scores,
)


class TestContextual(unittest.TestCase):

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
