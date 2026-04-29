"""Prompt expansion helpers used by manual_label.py.

This module provides a lightweight local replacement for historical prompt utilities.
It does not call any external LLM services.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

_PREFIX = "a picture of a dog "
_CUSTOM_PROMPTS: Dict[str, List[str]] = {
    "running": [
        "a dog running at full speed with legs extended",
        "overhead view of a dog sprinting rapidly across the floor with limbs spread wide",
        "a dog captured from the side running fast with motion blur on its legs",
        "a dog with all four legs fully stretched out mid-stride moving at high speed",
        "a dog in fast motion covering large distance across the room",
    ],
    "eating": [
        # D1: 靜態姿勢 — 頭低向碗 + 嘴接觸食物（提升安靜吃飯的召回）
        "a dog with its head lowered into a food bowl, mouth in contact with food",
        # D2: 俯拍靜態 — 頭朝下 + 碗的位置關係（不強制可見咀嚼）
        "overhead view of a dog with its snout pointed straight down into a food bowl while eating",
        # D3: 側面靜態 — 頸部前彎 + 嘴在碗內
        "a dog filmed from the side with its neck bent forward and snout inside a food bowl",
        # D4: 動態咀嚼（保留動作特徵 prompt，維持明顯咀嚼場景的匹配度）
        "a dog actively chewing food with jaw movement visible near a food bowl",
        # D5: 折中 — 姿勢+動作，明確不是 sitting idle
        "a dog leaning forward over a food bowl with body angled down, actively consuming food",
    ],
    "walking": [
        "a dog walking slowly with legs in alternating stepping motion",
        "overhead view of a dog moving at a calm steady pace across the floor",
        "a dog filmed from the side taking slow deliberate steps forward",
        "a dog with two diagonal legs lifted off the ground in a walking gait",
        "a dog casually strolling around the room with gradual displacement",
    ],
    "standing": [
        # D1: four legs on ground + body horizontal
        "a dog standing still with all four legs on the ground and body parallel to the floor",
        # D2: overhead — rectangular silhouette distinct from sitting triangle
        "overhead view of a dog stationary with an elongated rectangular body outline and all four legs visible",
        # D3: side — legs fully extended, belly off ground
        "a dog filmed from the side standing upright with all four legs fully extended and belly off the floor",
        # D4: weight evenly on all paws, spine level
        "a dog with all four paws evenly planted on the floor, spine level, and body fully upright",
        # D5: alert standing — distinguishes from lying and sitting
        "a dog alert and motionless in a standing posture with legs straight and head level",
    ],
    "sitting": [
        # D1: bottom-on-floor + head upright — contrasts with eating (head lowered)
        "a dog sitting with its bottom on the floor, front legs straight, and head held upright",
        # D2: 俯拍 — 強化「頭朝上 + 無食物互動」與 eating 區分
        "overhead view of a dog in a sitting position with head up and no food bowl interaction",
        # D3: side view — head above back level distinguishes from eating
        "a dog filmed from the side with hindquarters on the ground and head raised above its back",
        # D4: specific leg geometry — rear legs folded, front legs vertical
        "a dog with rear end on the floor, front legs vertical and upright, head erect, not interacting with food",
        # D5: resting-sit, body still, gaze forward
        "a dog seated quietly with ears alert, gaze forward, and no food bowl interaction",
    ],
    "lying": [
        "a dog lying down with its entire body flat on the floor",
        "overhead view of a dog in a recumbent posture with a round flat body silhouette",
        "a dog filmed from the side sprawled out on the ground with legs tucked",
        "a dog with belly touching the floor, legs folded under its body, and chin resting down",
        "a dog resting motionless on the ground with no limb movement",
    ],
}

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
    """Aggregate per-prompt scores into one score per base label.

    Notes
    -----
    SigLIP scores are normalized over the *flattened prompt list*. After
    reducing multiple prompts back to one column per base label, we normalize
    rows again so confidence thresholds remain on an interpretable 0..1 scale.
    """
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

    label_cols = list(prompt_map.keys())
    if label_cols:
        row_sum = out[label_cols].sum(axis=1)
        safe_sum = row_sum.where(row_sum > 0.0, 1.0)
        out.loc[:, label_cols] = out[label_cols].div(safe_sum, axis=0)

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


def classify_with_single_prompt(
    classifier,
    frames: Sequence,
    labels: Sequence[str],
    timestamps: Sequence[float] | None = None,
    camera_context: str = "",
) -> pd.DataFrame:
    """Run System-A style inference: one prompt per class (D1 variant only)."""
    result = build_label_prompt_result(
        labels=labels,
        mode="template",
        camera_context=camera_context,
    )
    single_map = {lbl: [variants[0]] for lbl, variants in result["prompt_map"].items()}
    flat_prompts = flatten_prompt_map(single_map)
    raw_scores = classifier.classify_frames(frames, flat_prompts, timestamps)
    return aggregate_prompt_scores(raw_scores, single_map, reducer="max")


def _template_prompts(label: str) -> List[str]:
    """Return custom prompt variants for a supported label."""
    custom = _CUSTOM_PROMPTS.get(label.strip().lower())
    if custom is not None:
        return list(custom)
    normalised = label.strip().lower()
    return [
        f"{_PREFIX}{normalised}",
        f"a dog {normalised} indoors",
        f"a dog {normalised} on the floor",
        f"overhead view of a dog {normalised}",
        f"a dog filmed from the side while {normalised}",
    ]


def _normalize_label(label: str) -> str:
    text = label.strip()
    if text.lower().startswith(_PREFIX):
        return text[len(_PREFIX) :].strip()
    return text
