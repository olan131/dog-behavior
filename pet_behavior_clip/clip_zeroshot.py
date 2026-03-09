"""clip_zeroshot.py – Zero-shot behaviour classification via SigLIP.

The module wraps the HuggingFace ``transformers`` SigLIP pipeline so that a
list of PIL images can be scored against an arbitrary set of text labels in a
single call. Similarity scores are converted to per-label probabilities
(sigmoid for SigLIP, softmax for CLIP), and the results are returned as a
``pandas.DataFrame``.

Responsibilities
----------------
* Load a SigLIP / CLIP model once and reuse it for multiple inference calls.
* Accept batches of PIL Images and return a DataFrame with columns for every
  behaviour label plus a ``timestamp`` column.
* Expose a convenience helper ``classify_video()`` that combines
  :class:`~pet_behavior_clip.video.VideoReader` with this classifier.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Default model – SigLIP is preferred; fall back to CLIP if unavailable.
_DEFAULT_MODEL = "google/siglip-so400m-patch14-224"
_FALLBACK_MODEL = "openai/clip-vit-base-patch32"


class SigLIPClassifier:
    """Zero-shot frame classifier backed by a SigLIP (or CLIP) model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the SigLIP SO-400M variant.
    device:
        ``"cpu"`` or ``"cuda"``.  Detected automatically when *None*.
    batch_size:
        Number of frames processed in a single forward pass.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._processor = None
        self._model = None
        self._model_type: str = "siglip"
        self._text_max_length: Optional[int] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model and processor (called lazily on first use)."""
        if self._model is not None:
            return

        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._load_siglip()
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not load SigLIP model %s (%s); falling back to CLIP %s.",
                self.model_name,
                exc,
                _FALLBACK_MODEL,
            )
            self.model_name = _FALLBACK_MODEL
            self._load_clip()

    def _load_siglip(self) -> None:
        from transformers import AutoProcessor, AutoModel

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._model_type = "siglip"
        self._text_max_length = self._infer_text_max_length()
        logger.info("Loaded SigLIP model: %s on %s", self.model_name, self._device)

    def _load_clip(self) -> None:
        from transformers import CLIPProcessor, CLIPModel

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._model_type = "clip"
        self._text_max_length = self._infer_text_max_length()
        logger.info("Loaded CLIP model: %s on %s", self.model_name, self._device)

    def _infer_text_max_length(self) -> Optional[int]:
        """Best-effort lookup for tokenizer/model text position limit."""
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer_limit = getattr(tokenizer, "model_max_length", None)
            if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
                return tokenizer_limit

        config = getattr(self._model, "config", None)
        text_cfg = getattr(config, "text_config", None) if config is not None else None
        for cfg in (text_cfg, config):
            if cfg is None:
                continue
            max_positions = getattr(cfg, "max_position_embeddings", None)
            if isinstance(max_positions, int) and max_positions > 0:
                return max_positions
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_frames(
        self,
        frames: Sequence[Image.Image],
        labels: Sequence[str],
        timestamps: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Score *frames* against each text *label*.

        Parameters
        ----------
        frames:
            Sequence of PIL Images (one per video sample).
        labels:
            Behaviour descriptions, e.g. ``["dog sitting", "dog running"]``.
        timestamps:
            Optional per-frame timestamp in seconds.

        Returns
        -------
        pandas.DataFrame
            Shape ``(len(frames), len(labels) + 1)``.
            Columns: ``timestamp`` + one column per label.
            Values are softmax probabilities in ``[0, 1]``.
        """
        self._load()

        if timestamps is None:
            timestamps = list(range(len(frames)))

        all_probs: List[np.ndarray] = []
        for i in range(0, len(frames), self.batch_size):
            batch_imgs = list(frames[i : i + self.batch_size])
            probs = self._score_batch(batch_imgs, list(labels))
            all_probs.append(probs)

        probs_matrix = np.vstack(all_probs)  # (N, L)
        df = pd.DataFrame(probs_matrix, columns=list(labels))
        df.insert(0, "timestamp", list(timestamps))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(
        self, images: List[Image.Image], labels: List[str]
    ) -> np.ndarray:
        """Return softmax probabilities for a batch of images. Shape: (B, L)."""
        import torch

        use_autocast = str(self._device).startswith("cuda")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_autocast):
                inputs = self._processor(
                    text=labels,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._text_max_length,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)

                if self._model_type == "siglip":
                    # Convert independent SigLIP scores to per-frame relative confidence
                    # so downstream percentages remain interpretable.
                    logits = outputs.logits_per_image  # (B, L)
                    probs = torch.sigmoid(logits)
                    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    probs = (probs / denom).cpu().numpy()
                else:
                    logits = outputs.logits_per_image  # (B, L)
                    probs = logits.softmax(dim=-1).cpu().numpy()

        return probs  # (B, L)
