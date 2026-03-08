"""Tests for video.py (VideoReader).

We avoid real video I/O by patching OpenCV so these tests run without any
media file or GPU present.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, frames: int = 10, fps: float = 25.0) -> None:
        self._frames = frames
        self._fps = fps
        self._pos = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        _CAP_PROP_FPS = 5
        _CAP_PROP_FRAME_COUNT = 7
        if prop == _CAP_PROP_FPS:
            return self._fps
        if prop == _CAP_PROP_FRAME_COUNT:
            return float(self._frames)
        return 0.0

    def set(self, prop: int, value: float) -> None:
        if prop == 1:  # CAP_PROP_POS_FRAMES
            self._pos = int(value)

    def read(self):
        import numpy as np

        if self._pos >= self._frames:
            return False, None
        frame = np.zeros((4, 4, 3), dtype="uint8")
        self._pos += 1
        return True, frame

    def release(self) -> None:
        pass


def _make_mock_cv2(n_frames: int = 10, fps: float = 25.0):
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.VideoCapture.return_value = _FakeCapture(n_frames, fps)
    mock_cv2.COLOR_BGR2RGB = 4

    def fake_cvt(img, code):
        return img

    mock_cv2.cvtColor.side_effect = fake_cvt
    return mock_cv2


class TestVideoReader(unittest.TestCase):

    def _make_reader(self, n_frames: int = 10, fps: float = 25.0, sample_fps: float = 1.0):
        """Return a VideoReader with patched OpenCV."""
        from unittest.mock import patch
        import sys

        mock_cv2 = _make_mock_cv2(n_frames, fps)

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            # Re-import video module with patched cv2
            if "pet_behavior_clip.video" in sys.modules:
                del sys.modules["pet_behavior_clip.video"]
            from pet_behavior_clip import video as vm

        return vm, mock_cv2

    def test_file_not_found(self):
        mock_cv2 = _make_mock_cv2()
        import sys

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            if "pet_behavior_clip.video" in sys.modules:
                del sys.modules["pet_behavior_clip.video"]
            from pet_behavior_clip import video as vm

            with self.assertRaises(FileNotFoundError):
                vm.VideoReader("/nonexistent/path/video.mp4")

    def test_sample_frames_count(self):
        """Sample 1 fps from a 25 fps, 25-frame video → ~1 frame (every 25th)."""
        import sys
        import tempfile, os

        # Create a fake file so FileNotFoundError is not raised
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            mock_cv2 = _make_mock_cv2(n_frames=25, fps=25.0)

            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                if "pet_behavior_clip.video" in sys.modules:
                    del sys.modules["pet_behavior_clip.video"]
                from pet_behavior_clip import video as vm

                reader = vm.VideoReader(tmp_path, sample_fps=1.0)
                frames = reader.sample_frames()
                # 25 frames at 25 fps = 1 second of video sampled at 1 fps → at least 1 frame
                self.assertGreaterEqual(len(frames), 1)
                ts, img = frames[0]
                self.assertIsInstance(ts, float)
        finally:
            os.unlink(tmp_path)

    def test_sample_frames_returns_pil(self):
        """Each frame in result is a (float, PIL.Image) tuple."""
        import sys, os, tempfile
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            mock_cv2 = _make_mock_cv2(n_frames=5, fps=5.0)

            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                if "pet_behavior_clip.video" in sys.modules:
                    del sys.modules["pet_behavior_clip.video"]
                from pet_behavior_clip import video as vm

                reader = vm.VideoReader(tmp_path, sample_fps=5.0)
                frames = reader.sample_frames()
                for ts, img in frames:
                    self.assertIsInstance(ts, float)
                    self.assertIsInstance(img, Image.Image)
        finally:
            os.unlink(tmp_path)

    def test_context_manager(self):
        """VideoReader should work as a context manager."""
        import sys, os, tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            mock_cv2 = _make_mock_cv2()

            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                if "pet_behavior_clip.video" in sys.modules:
                    del sys.modules["pet_behavior_clip.video"]
                from pet_behavior_clip import video as vm

                with vm.VideoReader(tmp_path) as reader:
                    frames = reader.sample_frames()
                self.assertIsInstance(frames, list)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
