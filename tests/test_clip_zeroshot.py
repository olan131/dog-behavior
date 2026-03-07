"""Tests for clip_zeroshot.py (SigLIPClassifier).

Uses a mock model/processor so no HuggingFace download is needed.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image


def _make_fake_transformers():
    """Return a mock transformers module with a minimal SigLIP implementation."""

    class FakeProcessor:
        def __call__(self, text, images, return_tensors, padding):
            import torch

            B = len(images)
            L = len(text)
            return {
                "input_ids": torch.zeros(L, 10, dtype=torch.long),
                "pixel_values": torch.zeros(B, 3, 224, 224),
                "attention_mask": torch.ones(L, 10, dtype=torch.long),
            }

    class FakeOutput:
        def __init__(self, B, L):
            import torch

            self.logits_per_image = torch.rand(B, L)

    class FakeModel:
        def __init__(self):
            self._device = "cpu"

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch

            B = kwargs["pixel_values"].shape[0]
            L = kwargs["input_ids"].shape[0]
            return FakeOutput(B, L)

        def parameters(self):
            import torch
            return iter([torch.tensor([0.0])])

    mock_tf = types.ModuleType("transformers")
    mock_tf.AutoProcessor = MagicMock()
    mock_tf.AutoProcessor.from_pretrained = MagicMock(return_value=FakeProcessor())
    mock_tf.AutoModel = MagicMock()
    mock_tf.AutoModel.from_pretrained = MagicMock(return_value=FakeModel())
    mock_tf.CLIPProcessor = MagicMock()
    mock_tf.CLIPModel = MagicMock()
    return mock_tf


class TestSigLIPClassifier(unittest.TestCase):

    def setUp(self):
        self.mock_tf = _make_fake_transformers()
        # Patch sys.modules so that 'import transformers' inside clip_zeroshot returns our mock
        self.patcher = patch.dict("sys.modules", {"transformers": self.mock_tf})
        self.patcher.start()
        # Reload the module with patched transformers
        if "pet_behavior_clip.clip_zeroshot" in sys.modules:
            del sys.modules["pet_behavior_clip.clip_zeroshot"]
        from pet_behavior_clip import clip_zeroshot

        self.module = clip_zeroshot

    def tearDown(self):
        self.patcher.stop()
        if "pet_behavior_clip.clip_zeroshot" in sys.modules:
            del sys.modules["pet_behavior_clip.clip_zeroshot"]

    def _make_frames(self, n: int = 4):
        return [Image.new("RGB", (224, 224), color=(i * 10, 0, 0)) for i in range(n)]

    def test_returns_dataframe(self):
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        frames = self._make_frames(4)
        labels = ["label_a", "label_b", "label_c"]
        df = clf.classify_frames(frames, labels)
        self.assertIsInstance(df, pd.DataFrame)

    def test_shape(self):
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        frames = self._make_frames(6)
        labels = ["sit", "run", "bark"]
        df = clf.classify_frames(frames, labels)
        self.assertEqual(df.shape, (6, 4))  # timestamp + 3 labels

    def test_columns(self):
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        frames = self._make_frames(3)
        labels = ["sit", "run"]
        df = clf.classify_frames(frames, labels)
        self.assertIn("timestamp", df.columns)
        for label in labels:
            self.assertIn(label, df.columns)

    def test_timestamp_when_provided(self):
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        frames = self._make_frames(3)
        labels = ["sit"]
        ts = [0.0, 1.0, 2.0]
        df = clf.classify_frames(frames, labels, timestamps=ts)
        self.assertListEqual(list(df["timestamp"]), ts)

    def test_values_in_range(self):
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        frames = self._make_frames(5)
        labels = ["a", "b"]
        df = clf.classify_frames(frames, labels)
        for col in labels:
            self.assertTrue((df[col] >= 0).all())
            self.assertTrue((df[col] <= 1).all())

    def test_batch_size_respected(self):
        """Ensure batching still returns correct number of rows."""
        clf = self.module.SigLIPClassifier(
            model_name="fake-model", device="cpu", batch_size=2
        )
        frames = self._make_frames(7)
        labels = ["x", "y"]
        df = clf.classify_frames(frames, labels)
        self.assertEqual(len(df), 7)


if __name__ == "__main__":
    unittest.main()
