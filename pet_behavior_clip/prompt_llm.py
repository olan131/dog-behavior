"""prompt_llm.py - LLM-augmented prompt generation for zero-shot classification.

This module helps expand coarse behaviour classes (for example: Active,
Resting, Eating/Drinking) into richer natural-language prompt variants that are
better aligned with surveillance-camera imagery.

Three modes are supported:

- off: Use original labels only.
- template: Deterministic local prompt expansion (no API call).
- llm: Ask OpenRouter to generate prompt variants, with automatic fallback to
  template mode when API access is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Literal, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PromptMode = Literal["off", "template", "llm"]
PromptAggregate = Literal["max", "mean"]

_DEFAULT_PROMPT_MODEL = "openai/gpt-4o-mini"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_CAMERA_CONTEXT = (
    "indoor pet camera"
)
_DEFAULT_VARIANTS = 5
_MAX_PROMPT_WORDS = 8

_SYSTEM_PROMPT = (
    "You generate SigLIP-friendly zero-shot prompts for dog behavior recognition. "
    "Use short phrases with only visible actions/postures, avoid intent/emotion/health diagnosis, "
    "and keep each prompt <= 8 words. Return strict JSON only."
)


def build_label_prompt_map(
    labels: Sequence[str],
    mode: PromptMode = "template",
    camera_context: str = _DEFAULT_CAMERA_CONTEXT,
    llm_model: str = _DEFAULT_PROMPT_MODEL,
    llm_variants: int = _DEFAULT_VARIANTS,
    llm_api_key: str | None = None,
) -> Dict[str, List[str]]:
    """Backward-compatible wrapper that returns prompt map only."""
    result = build_label_prompt_result(
        labels=labels,
        mode=mode,
        camera_context=camera_context,
        llm_model=llm_model,
        llm_variants=llm_variants,
        llm_api_key=llm_api_key,
    )
    return result["prompt_map"]


def build_label_prompt_result(
    labels: Sequence[str],
    mode: PromptMode = "template",
    camera_context: str = _DEFAULT_CAMERA_CONTEXT,
    llm_model: str = _DEFAULT_PROMPT_MODEL,
    llm_variants: int = _DEFAULT_VARIANTS,
    llm_api_key: str | None = None,
) -> Dict[str, object]:
    """Build a mapping: class label -> list of prompt variants.

    In ``off`` mode, each label maps to itself.
    In ``template`` mode, deterministic variants are generated locally.
    In ``llm`` mode, prompts are generated via OpenRouter with template fallback.
    """
    cleaned = [l.strip() for l in labels if l and l.strip()]
    if not cleaned:
        return {
            "prompt_map": {},
            "source": "empty",
            "fallback_used": False,
            "fallback_reason": None,
        }

    if mode == "off":
        return {
            "prompt_map": {label: [label] for label in cleaned},
            "source": "off",
            "fallback_used": False,
            "fallback_reason": None,
        }

    if mode == "template":
        return {
            "prompt_map": _template_prompt_map(cleaned, camera_context),
            "source": "template",
            "fallback_used": False,
            "fallback_reason": None,
        }

    if mode == "llm":
        api_key = llm_api_key or _resolve_llm_api_key()
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set; using template prompt expansion.")
            return {
                "prompt_map": _template_prompt_map(cleaned, camera_context),
                "source": "template_fallback",
                "fallback_used": True,
                "fallback_reason": "OPENROUTER_API_KEY not set",
            }
        try:
            return {
                "prompt_map": _llm_prompt_map(
                labels=cleaned,
                camera_context=camera_context,
                model=llm_model,
                api_key=api_key,
                n_variants=llm_variants,
                ),
                "source": "llm",
                "fallback_used": False,
                "fallback_reason": None,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("LLM prompt generation failed (%s); using template mode.", exc)
            return {
                "prompt_map": _template_prompt_map(cleaned, camera_context),
                "source": "template_fallback",
                "fallback_used": True,
                "fallback_reason": str(exc),
            }

    raise ValueError(f"Unsupported prompt mode: {mode}")


def flatten_prompt_map(prompt_map: Dict[str, Sequence[str]]) -> List[str]:
    """Flatten prompt map into a de-duplicated prompt list preserving order."""
    seen = set()
    flat: List[str] = []
    for prompts in prompt_map.values():
        for p in prompts:
            key = p.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            flat.append(key)
    return flat


def aggregate_prompt_scores(
    prompt_scores: pd.DataFrame,
    prompt_map: Dict[str, Sequence[str]],
    reducer: PromptAggregate = "max",
) -> pd.DataFrame:
    """Aggregate prompt-level scores back to class-level scores.

    Parameters
    ----------
    prompt_scores:
        DataFrame with columns ``timestamp`` + prompt columns.
    prompt_map:
        Mapping from target class label to prompt variants used in inference.
    reducer:
        ``max`` or ``mean`` across prompt variants per class.
    """
    if "timestamp" not in prompt_scores.columns:
        raise ValueError("prompt_scores must include a 'timestamp' column")

    out = pd.DataFrame({"timestamp": prompt_scores["timestamp"]})

    for label, prompts in prompt_map.items():
        existing = [p for p in prompts if p in prompt_scores.columns]
        if not existing:
            out[label] = 0.0
            continue
        if reducer == "max":
            out[label] = prompt_scores[existing].max(axis=1)
        elif reducer == "mean":
            out[label] = prompt_scores[existing].mean(axis=1)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")

    return out


def _template_prompt_map(labels: Sequence[str], camera_context: str) -> Dict[str, List[str]]:
    prompt_map: Dict[str, List[str]] = {}
    for label in labels:
        prompt_map[label] = _template_variants_for_label(label, camera_context)
    return prompt_map


def _template_variants_for_label(label: str, camera_context: str) -> List[str]:
    """Generate deterministic prompt variants for one label."""
    key = label.strip().lower()

    exact_prompt_labels = {
        "a picture of an animal moving",
        "a picture of an animal eating",
        "a picture of an animal resting",
    }
    if key in exact_prompt_labels:
        return [key]

    active_keys = {"active", "activity", "moving", "play"}
    resting_keys = {"resting", "rest", "sleep", "calm", "idle"}
    eat_keys = {"eating/drinking", "eating", "drinking", "feeding"}
    moving_keys = {"dog moving", "moving", "dog in motion"}
    stationary_keys = {"dog stationary", "stationary", "dog not moving"}
    sitting_keys = {"dog sitting calmly", "dog sitting", "sitting"}
    walking_keys = {"dog walking normally", "dog walking", "walking"}
    running_keys = {"dog running", "running"}
    lying_keys = {"dog lying down", "dog lying", "lying down"}

    if key in moving_keys:
        actions = [
            "dog body translating",
            "dog stepping forward",
            "dog changing location",
            "dog moving across floor",
            "dog in continuous motion",
        ]
    elif key in stationary_keys:
        actions = [
            "dog staying in place",
            "dog no body translation",
            "dog static posture",
            "dog fixed location",
            "dog nearly motionless",
        ]
    elif key in sitting_keys:
        actions = [
            "dog sitting posture",
            "dog hindquarters on floor",
            "dog upright seated",
            "dog seated still",
            "dog sitting in place",
        ]
    elif key in walking_keys:
        actions = [
            "dog walking",
            "dog slow locomotion",
            "dog stepping steadily",
            "dog moving at walking pace",
            "dog forward gait",
        ]
    elif key in running_keys:
        actions = [
            "dog running",
            "dog fast locomotion",
            "dog rapid stride",
            "dog moving quickly",
            "dog sprinting movement",
        ]
    elif key in lying_keys:
        actions = [
            "dog lying down",
            "dog body on floor",
            "dog low resting posture",
            "dog recumbent posture",
            "dog lying still",
        ]
    elif key in active_keys:
        actions = [
            "dog running",
            "dog walking",
            "dog trotting",
            "dog changing position",
            "dog moving quickly",
        ]
    elif key in resting_keys:
        actions = [
            "dog lying down",
            "dog resting",
            "dog sleeping",
            "dog still posture",
            "dog low movement",
        ]
    elif key in eat_keys:
        actions = [
            "dog eating from bowl",
            "dog drinking water",
            "dog head near bowl",
            "dog licking near bowl",
            "dog chewing at bowl",
        ]
    else:
        actions = [
            f"dog {label}",
            f"dog posture {label}",
            f"dog action {label}",
            f"dog visible {label}",
            f"dog behavior {label}",
        ]

    return [_siglip_safe_prompt(a, camera_context) for a in actions]


def _siglip_safe_prompt(action_text: str, camera_context: str) -> str:
    """Normalize to a short, observable phrase suitable for SigLIP text encoder."""
    cleaned_action = re.sub(r"\s+", " ", action_text.strip().lower())
    cleaned_context = re.sub(r"\s+", " ", camera_context.strip().lower())
    merged = f"{cleaned_action} in {cleaned_context}" if cleaned_context else cleaned_action
    merged = re.sub(r"[^a-z0-9\s/-]", "", merged)

    words = [w for w in merged.split(" ") if w]
    if len(words) > _MAX_PROMPT_WORDS:
        words = words[:_MAX_PROMPT_WORDS]
    return " ".join(words)


def _llm_prompt_map(
    labels: Sequence[str],
    camera_context: str,
    model: str,
    api_key: str,
    n_variants: int,
) -> Dict[str, List[str]]:  # pragma: no cover - network dependent
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)

    user_prompt = (
        "Generate SigLIP-friendly zero-shot prompt variants for dog behavior recognition.\n"
        f"Labels: {list(labels)}\n"
        f"Camera context: {camera_context}\n"
        f"Variants per label: {n_variants}\n"
        "Constraints: each prompt <= 8 words; only visible actions/postures; no intent, emotion, or diagnosis; no punctuation-heavy sentences.\n"
        "Return JSON only in this exact schema:\n"
        '{"labels": [{"name": "<label>", "prompts": ["...", "..."]}]}'
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    content = _extract_content(resp)
    payload = json.loads(content)

    mapping: Dict[str, List[str]] = {}
    for item in payload.get("labels", []):
        name = str(item.get("name", "")).strip()
        prompts = [
            _siglip_safe_prompt(str(p), "")
            for p in item.get("prompts", [])
            if str(p).strip()
        ]
        if name:
            mapping[name] = prompts

    # Ensure all labels exist and are non-empty
    for label in labels:
        if label not in mapping or not mapping[label]:
            mapping[label] = _template_variants_for_label(label, camera_context)

    return mapping


def _extract_content(response) -> str:
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("OpenRouter response is empty")
    return response.choices[0].message.content.strip()


def _resolve_llm_api_key() -> str | None:
    """Resolve API key for OpenRouter only."""
    return os.getenv("OPENROUTER_API_KEY")
