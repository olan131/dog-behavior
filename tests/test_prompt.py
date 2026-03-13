"""Tests for prompt.py aggregation behavior."""

from __future__ import annotations

import unittest

import pandas as pd

from pet_behavior_clip.prompt import aggregate_prompt_scores, build_label_prompt_result


class TestAggregatePromptScores(unittest.TestCase):

    def test_max_reducer_renormalizes_label_scores(self):
        raw = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                "p_move_1": [0.10, 0.15],
                "p_move_2": [0.09, 0.12],
                "p_rest_1": [0.05, 0.04],
                "p_rest_2": [0.03, 0.02],
            }
        )
        prompt_map = {
            "moving": ["p_move_1", "p_move_2"],
            "resting": ["p_rest_1", "p_rest_2"],
        }

        out = aggregate_prompt_scores(raw, prompt_map, reducer="max")

        self.assertAlmostEqual(float(out.loc[0, "moving"] + out.loc[0, "resting"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[1, "moving"] + out.loc[1, "resting"]), 1.0, places=6)
        self.assertGreater(out.loc[0, "moving"], out.loc[0, "resting"])

    def test_missing_prompts_keep_zero_weight(self):
        raw = pd.DataFrame(
            {
                "timestamp": [0.0],
                "p_move": [0.2],
            }
        )
        prompt_map = {
            "moving": ["p_move"],
            "eating": ["p_eat_missing"],
        }

        out = aggregate_prompt_scores(raw, prompt_map, reducer="max")

        self.assertAlmostEqual(float(out.loc[0, "moving"]), 1.0, places=6)
        self.assertAlmostEqual(float(out.loc[0, "eating"]), 0.0, places=6)


class TestPromptTemplates(unittest.TestCase):

    def test_running_uses_custom_prompt_set(self):
        result = build_label_prompt_result(["running"], mode="template")
        prompts = result["prompt_map"]["running"]

        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0], "a dog running at full speed with legs extended")
        self.assertIn("motion blur on its legs", prompts[2])

    def test_unknown_label_falls_back_to_generic_prompts(self):
        result = build_label_prompt_result(["jumping"], mode="template")
        prompts = result["prompt_map"]["jumping"]
        self.assertEqual(len(prompts), 5)
        self.assertTrue(all("jumping" in p for p in prompts))


if __name__ == "__main__":
    unittest.main()
