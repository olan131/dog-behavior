"""Prompt expansion helpers used by manual_label.py.

This module provides a lightweight local replacement for historical prompt utilities.
It does not call any external LLM services.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

_PREFIX = "a picture of an animal "


def build_label_prompt_result(
    labels: Iterable[str],
    mode: str = "template",
    camera_context: str = "",
) -> dict:
    """Build a deterministic prompt map for each base label.

    Parameters
    ----------
    labels:
        Base behavior labels, e.g. "active" or "resting".
    mode:
        Supported: "template" only.
    camera_context:
        Optional context phrase prepended to each prompt.
    """
    if mode != "template":
        raise ValueError(f"Unsupported mode '{mode}'. Only 'template' is supported.")

    ctx = camera_context.strip()
    labels_list = [_normalize_label(str(label)) for label in labels if str(label).strip()]

    prompt_map: Dict[str, List[str]] = {}
    for label in labels_list:
        variants = _template_prompts(label)
        if ctx:
            variants = [f"{ctx}, {prompt}" for prompt in variants]
        prompt_map[label] = variants

    return {
        "mode": mode,
        "camera_context": ctx,
        "prompt_map": prompt_map,
    }


def flatten_prompt_map(prompt_map: dict[str, list[str]]) -> list[str]:
    """Flatten prompt map values into a stable prompt list."""
    prompts: List[str] = []
    for _, label_prompts in prompt_map.items():
        prompts.extend(label_prompts)
    return prompts


def aggregate_prompt_scores(
    raw_scores: pd.DataFrame,
    prompt_map: dict[str, list[str]],
    reducer: str = "max",
) -> pd.DataFrame:
    """Aggregate per-prompt scores into one score per base label."""
    if reducer not in {"max", "mean"}:
        raise ValueError(f"Unsupported reducer '{reducer}'. Use 'max' or 'mean'.")

    out = pd.DataFrame(index=raw_scores.index)
    if "timestamp" in raw_scores.columns:
        out["timestamp"] = raw_scores["timestamp"]

    for label, prompts in prompt_map.items():
        available = [p for p in prompts if p in raw_scores.columns]
        if not available:
            out[label] = 0.0
            continue

        block = raw_scores[available]
        if reducer == "max":
            out[label] = block.max(axis=1)
        else:
            out[label] = block.mean(axis=1)

    return out


def classify_with_template_max(
    classifier,
    frames: Sequence,
    labels: Sequence[str],
    timestamps: Sequence[float] | None = None,
    camera_context: str = "",
) -> pd.DataFrame:
    """Run System-B style inference: template prompts + max aggregation."""
    result = build_label_prompt_result(
        labels=labels,
        mode="template",
        camera_context=camera_context,
    )
    prompt_map = result["prompt_map"]
    flat_prompts = flatten_prompt_map(prompt_map)
    raw_scores = classifier.classify_frames(frames, flat_prompts, timestamps)
    return aggregate_prompt_scores(raw_scores, prompt_map, reducer="max")


def _template_prompts(label: str) -> List[str]:
    """Create 5 deterministic prompt variants for a label."""
    return [
        f"a picture of an animal {label}",
        f"an animal that is {label}",
        f"a surveillance view of an animal {label}",
        f"a top-down camera view of an animal {label}",
        f"pet behavior: {label}",
    ]


def _normalize_label(label: str) -> str:
    text = label.strip()
    if text.lower().startswith(_PREFIX):
        return text[len(_PREFIX) :].strip()
    return text
