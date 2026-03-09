"""Tests for behavior_postprocess.py."""

from __future__ import annotations

import unittest

import pandas as pd

from pet_behavior_clip.behavior_postprocess import (
    build_behavior_segments,
    infer_frame_behaviors,
    smooth_behavior_labels,
)


class TestInferFrameBehaviors(unittest.TestCase):

    def test_anomaly_has_priority_over_scores(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0],
                "a picture of an animal moving": [0.9],
                "a picture of an animal eating": [0.05],
                "a picture of an animal resting": [0.05],
                "anomaly_score": [3.2],
                "is_anomaly": [True],
            }
        )
        out = infer_frame_behaviors(df, confidence_threshold=0.45)
        self.assertEqual(out.loc[0, "behavior_label"], "anomaly")

    def test_uncertain_when_max_score_below_threshold(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0],
                "a picture of an animal moving": [0.33],
                "a picture of an animal eating": [0.40],
                "a picture of an animal resting": [0.27],
                "anomaly_score": [0.9],
                "is_anomaly": [False],
            }
        )
        out = infer_frame_behaviors(df, confidence_threshold=0.45)
        self.assertEqual(out.loc[0, "behavior_label"], "uncertain")


class TestSmoothAndSegments(unittest.TestCase):

    def test_majority_vote_smoothing(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
                "behavior_label": ["resting", "moving", "resting", "resting", "resting"],
            }
        )
        out = smooth_behavior_labels(df, window_seconds=1.0)
        self.assertEqual(out.loc[1, "behavior_smooth"], "resting")

    def test_anomaly_is_fixed_during_smoothing(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "behavior_label": ["resting", "anomaly", "resting"],
            }
        )
        out = smooth_behavior_labels(df, window_seconds=2.0)
        self.assertEqual(out.loc[1, "behavior_smooth"], "anomaly")

    def test_segment_merge_contiguous_labels(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
                "behavior_smooth": ["anomaly", "anomaly", "resting", "resting", "moving"],
            }
        )
        segments = build_behavior_segments(df, label_col="behavior_smooth")
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments.loc[0, "start_s"], 0.0)
        self.assertEqual(segments.loc[0, "end_s"], 1.0)
        self.assertEqual(segments.loc[0, "label"], "anomaly")


if __name__ == "__main__":
    unittest.main()
