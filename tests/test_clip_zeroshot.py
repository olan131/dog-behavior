"""Tests for clip_zeroshot.py (SigLIPClassifier).

Uses a mock model/processor so no HuggingFace download or PyTorch is needed.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Shared fake tensor
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Minimal stand-in for torch.Tensor that supports .cpu().numpy()."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self._data.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def softmax(self, dim):
        e = np.exp(self._data - self._data.max(axis=dim, keepdims=True))
        return _FakeTensor(e / e.sum(axis=dim, keepdims=True))

    def to(self, device):
        return self

    def __add__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._data + other._data)
        return _FakeTensor(self._data + np.asarray(other))

    __radd__ = __add__


# ---------------------------------------------------------------------------
# Fake torch module
# ---------------------------------------------------------------------------

class _NoGradCtx:
    """Context manager stand-in for torch.no_grad()."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass


def _make_fake_torch():
    mock_torch = types.ModuleType("torch")
    mock_torch.no_grad = _NoGradCtx
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.sigmoid = lambda t: _FakeTensor(
        1.0 / (1.0 + np.exp(-t._data if isinstance(t, _FakeTensor) else np.asarray(t)))
    )
    mock_torch.Tensor = _FakeTensor
    return mock_torch


# ---------------------------------------------------------------------------
# Fake transformers module
# ---------------------------------------------------------------------------

def _make_fake_transformers():
    class FakeProcessor:
        def __call__(self, text, images, return_tensors, padding):
            B = len(images)
            L = len(text)
            return {
                "input_ids": _FakeTensor(np.zeros((L, 10), dtype=np.int64)),
                "pixel_values": _FakeTensor(np.zeros((B, 3, 224, 224), dtype=np.float32)),
                "attention_mask": _FakeTensor(np.ones((L, 10), dtype=np.int64)),
            }

    class FakeOutput:
        def __init__(self, B, L):
            self.logits_per_image = _FakeTensor(
                np.random.rand(B, L).astype(np.float32)
            )

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            B = kwargs["pixel_values"].shape[0]
            L = kwargs["input_ids"].shape[0]
            return FakeOutput(B, L)

    mock_tf = types.ModuleType("transformers")
    mock_tf.AutoProcessor = MagicMock()
    mock_tf.AutoProcessor.from_pretrained = MagicMock(return_value=FakeProcessor())
    mock_tf.AutoModel = MagicMock()
    mock_tf.AutoModel.from_pretrained = MagicMock(return_value=FakeModel())
    mock_tf.CLIPProcessor = MagicMock()
    mock_tf.CLIPModel = MagicMock()
    return mock_tf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSigLIPClassifier(unittest.TestCase):

    def setUp(self):
        self.mock_torch = _make_fake_torch()
        self.mock_tf = _make_fake_transformers()
        self.patcher = patch.dict(
            "sys.modules",
            {
                "transformers": self.mock_tf,
                "torch": self.mock_torch,
            },
        )
        self.patcher.start()
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

    def test_siglip_applies_model_logit_bias(self):
        """SigLIP path should add model.logit_bias before sigmoid."""
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")

        class _BiasModel:
            def __init__(self):
                self.logit_bias = _FakeTensor(np.array([[2.0, -2.0]], dtype=np.float32))

            def __call__(self, **kwargs):
                return outputs

        clf._model = _BiasModel()
        clf._model_type = "siglip"

        outputs = types.SimpleNamespace(
            logits_per_image=_FakeTensor(np.array([[0.0, 0.0]], dtype=np.float32))
        )
        clf._processor = MagicMock(return_value={"dummy": _FakeTensor(np.array([1.0]))})
        probs = clf._score_batch([Image.new("RGB", (224, 224))], ["a", "b"])

        self.assertGreater(probs[0, 0], 0.5)
        self.assertLess(probs[0, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
