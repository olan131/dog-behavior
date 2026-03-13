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

    def sum(self, dim=None, keepdim=False):
        axis = dim
        return _FakeTensor(self._data.sum(axis=axis, keepdims=keepdim))

    def clamp_min(self, value):
        return _FakeTensor(np.maximum(self._data, value))

    def exp(self):
        return _FakeTensor(np.exp(self._data))

    @property
    def T(self):
        return _FakeTensor(self._data.T)

    def __matmul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._data @ other._data)
        return _FakeTensor(self._data @ np.asarray(other))

    def __mul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._data * other._data)
        return _FakeTensor(self._data * np.asarray(other))

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._data + other._data)
        return _FakeTensor(self._data + np.asarray(other))

    __radd__ = __add__

    def norm(self, dim=None, keepdim=False):
        return _FakeTensor(
            np.linalg.norm(self._data, axis=dim, keepdims=keepdim)
        )

    def __truediv__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._data / other._data)
        return _FakeTensor(self._data / np.asarray(other))


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
    mock_torch.inference_mode = _NoGradCtx
    mock_torch.autocast = lambda device_type, enabled=False: _NoGradCtx()
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
        def __call__(self, text, images, return_tensors, padding, truncation=False, max_length=None):
            result = {}
            if text is not None:
                L = len(text)
                result["input_ids"] = _FakeTensor(np.zeros((L, 10), dtype=np.int64))
                result["attention_mask"] = _FakeTensor(np.ones((L, 10), dtype=np.int64))
            if images is not None:
                B = len(images)
                result["pixel_values"] = _FakeTensor(np.zeros((B, 3, 224, 224), dtype=np.float32))
            return result

    class FakeOutput:
        def __init__(self, B, L):
            self.logits_per_image = _FakeTensor(
                np.random.rand(B, L).astype(np.float32)
            )

    class FakeModel:
        def __init__(self):
            # exp(0) = 1.0 — neutral scale so dot-product magnitudes are preserved
            self.logit_scale = _FakeTensor(np.array(0.0, dtype=np.float32))
            self.config = None

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            B = kwargs["pixel_values"].shape[0]
            L = kwargs["input_ids"].shape[0]
            return FakeOutput(B, L)

        def get_text_features(self, **kwargs):
            L = kwargs["input_ids"].shape[0]
            return _FakeTensor(np.ones((L, 8), dtype=np.float32) / np.sqrt(8))

        def get_image_features(self, **kwargs):
            B = kwargs["pixel_values"].shape[0]
            return _FakeTensor(np.ones((B, 8), dtype=np.float32) / np.sqrt(8))

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

    def test_logit_bias_applied_when_present(self):
        """When model has logit_bias, it should shift the output probabilities."""
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        clf._load()
        clf._model_type = "siglip"
        # Strong bias: label A heavily favoured, label B suppressed
        clf._model.logit_bias = _FakeTensor(np.array([[10.0, -10.0]], dtype=np.float32))

        text_features = _FakeTensor(np.ones((2, 8), dtype=np.float32) / np.sqrt(8))
        probs = clf._score_batch([Image.new("RGB", (224, 224))], text_features)

        self.assertGreater(float(probs[0, 0]), float(probs[0, 1]))

    def test_encode_text_called_with_truncation(self):
        """_encode_text should pass truncation=True and max_length to the processor."""
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        clf._load()

        proc_calls = {}

        def _proc(*, text, images, return_tensors, padding, truncation=False, max_length=None):
            proc_calls["truncation"] = truncation
            proc_calls["max_length"] = max_length
            return {
                "input_ids": _FakeTensor(np.zeros((len(text), 10), dtype=np.int64)),
                "attention_mask": _FakeTensor(np.ones((len(text), 10), dtype=np.int64)),
            }

        clf._processor = _proc
        clf._text_max_length = 16
        clf._encode_text(["long long long prompt"])

        self.assertTrue(proc_calls.get("truncation"))
        self.assertEqual(proc_calls.get("max_length"), 16)

    def test_siglip_outputs_are_row_normalized(self):
        """SigLIP outputs should sum to ~1 across labels for each frame."""
        clf = self.module.SigLIPClassifier(model_name="fake-model", device="cpu")
        clf._load()
        clf._model_type = "siglip"

        # Pre-encoded text features for 3 labels, dimension 8
        text_features = _FakeTensor(np.ones((3, 8), dtype=np.float32) / np.sqrt(8))
        probs = clf._score_batch(
            [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))],
            text_features,
        )

        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
