"""video.py – Video reading and frame sampling utilities.

Responsibilities
----------------
* Open a video file using OpenCV.
* Sample frames at a user-specified rate (frames per second).
* Return frames as PIL Images ready to be consumed by a vision model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Tuple

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class VideoReader:
    """Read a video file and expose its frames as PIL Images.

    Parameters
    ----------
    path:
        Path to the video file.
    sample_fps:
        How many frames to sample per second of video.  Defaults to ``1``.
    """

    def __init__(self, path: str | Path, sample_fps: float = 1.0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.sample_fps = sample_fps
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def native_fps(self) -> float:
        """Return the native frame-rate of the video."""
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 25.0

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        cap = self._open()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_seconds(self) -> float:
        """Duration of the video in seconds."""
        native = self.native_fps
        return self.frame_count / native if native > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_frames(self) -> List[Tuple[float, Image.Image]]:
        """Return a list of ``(timestamp_seconds, PIL_Image)`` tuples.

        Frames are sampled at *sample_fps* from the source video.
        """
        return list(self._iter_frames())

    def _iter_frames(self) -> Generator[Tuple[float, Image.Image], None, None]:
        cap = self._open()
        native = self.native_fps
        if self.sample_fps > native:
            logger.warning(
                "Requested sample_fps (%.2f) exceeds native fps (%.2f); "
                "clamping to native fps.",
                self.sample_fps,
                native,
            )
        step = max(1, round(native / self.sample_fps))

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                timestamp = frame_idx / native
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                yield timestamp, pil_img
            frame_idx += 1

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying OpenCV capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video: {self.path}")
        return self._cap
