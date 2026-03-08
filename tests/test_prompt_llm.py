"""Tests for prompt_llm.py."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pet_behavior_clip.prompt_llm import (
    aggregate_prompt_scores,
    build_label_prompt_map,
    build_label_prompt_result,
    flatten_prompt_map,
)


class TestPromptLLM(unittest.TestCase):

    def test_off_mode_maps_to_original_labels(self):
        labels = ["Active", "Resting", "Eating/Drinking"]
        prompt_map = build_label_prompt_map(labels, mode="off")
        self.assertEqual(prompt_map["Active"], ["Active"])
        self.assertEqual(prompt_map["Resting"], ["Resting"])
        self.assertEqual(prompt_map["Eating/Drinking"], ["Eating/Drinking"])

    def test_template_mode_generates_multiple_prompts(self):
        labels = ["Active", "Resting", "Eating/Drinking"]
        prompt_map = build_label_prompt_map(labels, mode="template")
        for label in labels:
            self.assertIn(label, prompt_map)
            self.assertGreaterEqual(len(prompt_map[label]), 5)

    def test_llm_mode_without_key_exposes_fallback_result(self):
        labels = ["Active", "Resting"]
        with patch.dict("os.environ", {}, clear=True):
            result = build_label_prompt_result(labels, mode="llm")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["source"], "template_fallback")
        self.assertIn("OPENROUTER_API_KEY not set", str(result["fallback_reason"]))
        self.assertIn("Active", result["prompt_map"])

    def test_build_label_prompt_map_stays_backward_compatible(self):
        labels = ["Active", "Resting"]
        with patch.dict("os.environ", {}, clear=True):
            prompt_map = build_label_prompt_map(labels, mode="llm")
        self.assertIsInstance(prompt_map, dict)
        self.assertIn("Active", prompt_map)

    def test_flatten_prompt_map_deduplicates(self):
        prompt_map = {
            "A": ["p1", "p2"],
            "B": ["p2", "p3"],
        }
        flat = flatten_prompt_map(prompt_map)
        self.assertEqual(flat, ["p1", "p2", "p3"])

    def test_aggregate_prompt_scores_max(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                "a1": [0.1, 0.7],
                "a2": [0.6, 0.4],
                "r1": [0.3, 0.2],
                "e1": [0.4, 0.8],
            }
        )
        prompt_map = {
            "Active": ["a1", "a2"],
            "Resting": ["r1"],
            "Eating/Drinking": ["e1"],
        }
        out = aggregate_prompt_scores(df, prompt_map, reducer="max")
        self.assertListEqual(list(out.columns), ["timestamp", "Active", "Resting", "Eating/Drinking"])
        self.assertAlmostEqual(out.loc[0, "Active"], 0.6)
        self.assertAlmostEqual(out.loc[1, "Active"], 0.7)

    def test_aggregate_prompt_scores_mean(self):
        df = pd.DataFrame(
            {
                "timestamp": [0.0],
                "a1": [0.2],
                "a2": [0.8],
            }
        )
        prompt_map = {"Active": ["a1", "a2"]}
        out = aggregate_prompt_scores(df, prompt_map, reducer="mean")
        self.assertAlmostEqual(out.loc[0, "Active"], 0.5)


if __name__ == "__main__":
    unittest.main()
