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
from typing import Dict, List, Literal, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PromptMode = Literal["off", "template", "llm"]
PromptAggregate = Literal["max", "mean"]

_DEFAULT_PROMPT_MODEL = "openai/gpt-4o-mini"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_CAMERA_CONTEXT = (
    "4K overhead fisheye surveillance camera view inside a pet living space"
)
_DEFAULT_VARIANTS = 5

_SYSTEM_PROMPT = (
    "You generate robust zero-shot text prompts for vision-language models. "
    "Return strict JSON only. Keep prompts concise and visual."
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

    active_keys = {"active", "activity", "moving", "play"}
    resting_keys = {"resting", "rest", "sleep", "calm", "idle"}
    eat_keys = {"eating/drinking", "eating", "drinking", "feeding"}
    sit_keys = {"sitting", "sit"}
    walk_keys = {"walking", "walk", "walking normally"}
    run_keys = {"running", "run"}
    bark_keys = {"barking", "bark"}
    limp_keys = {"limping", "limp"}
    scratch_keys = {"scratching", "scratch"}
    shake_keys = {"shaking", "shake", "trembling", "tremble"}
    lying_keys = {"lying down", "lying", "laid down"}

    if key in active_keys:
        actions = [
            "dog is moving actively",
            "dog is walking or trotting",
            "dog is running or changing position quickly",
            "dog is alert and exploring the area",
            "dog shows sustained locomotion",
        ]
    elif key in resting_keys:
        actions = [
            "dog is resting",
            "dog is lying down calmly",
            "dog is sleeping or nearly motionless",
            "dog remains in one place with low activity",
            "dog posture indicates relaxation",
        ]
    elif key in eat_keys:
        actions = [
            "dog is eating from a bowl",
            "dog is drinking water",
            "dog head is lowered at food or water station",
            "dog repeatedly licks or chews near the bowl",
            "dog is engaged in feeding behavior",
        ]
    elif key in sit_keys:
        actions = [
            "dog is sitting still",
            "dog is in a seated position with hindquarters on the ground",
            "dog sitting upright and alert",
            "dog sits calmly in place",
            "dog adopts a still, seated posture",
        ]
    elif key in walk_keys:
        actions = [
            "dog is walking at a normal pace",
            "dog moves steadily with all four legs in regular rhythm",
            "dog walking with even, balanced gait",
            "dog strolls without any sign of discomfort",
            "dog locomotion appears smooth and symmetrical",
        ]
    elif key in run_keys:
        actions = [
            "dog is running quickly",
            "dog gallops or sprints across the area",
            "dog moves at high speed with extended strides",
            "dog is dashing or chasing",
            "dog runs with energetic, rapid movement",
        ]
    elif key in bark_keys:
        actions = [
            "dog is barking with mouth open",
            "dog vocalises repeatedly with open mouth",
            "dog stands alert and barks at something",
            "dog displays barking posture with raised head",
            "dog shows agitated barking behavior",
        ]
    elif key in limp_keys:
        actions = [
            "dog is limping and favouring one leg",
            "dog walks with an uneven, irregular gait",
            "dog shows lameness by lifting or sparing one paw",
            "dog movement is asymmetrical due to leg pain",
            "dog exhibits abnormal weight-bearing while walking",
        ]
    elif key in scratch_keys:
        actions = [
            "dog is scratching itself with its paw",
            "dog repeatedly scratches an area of its body",
            "dog rubs or scratches head, ear, or flank",
            "dog displays itching behaviour with vigorous scratching",
            "dog pauses movement to scratch persistently",
        ]
    elif key in shake_keys:
        actions = [
            "dog is shaking or trembling its body",
            "dog shakes its whole body rapidly",
            "dog shows repetitive trembling motion",
            "dog quivers or vibrates unusually",
            "dog exhibits uncontrolled body shaking",
        ]
    elif key in lying_keys:
        actions = [
            "dog is lying flat on the ground",
            "dog lies down with body fully extended",
            "dog is in a prone position on the floor",
            "dog rests stretched out on its side or belly",
            "dog adopts a flat, recumbent posture",
        ]
    else:
        actions = [
            f"dog behavior category: {label}",
            f"surveillance view of dog: {label}",
            f"top-down pet camera scene with dog {label}",
            f"dog appears to be {label}",
            f"canine posture consistent with {label}",
        ]

    return [f"{a}, {camera_context}" for a in actions]


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
        "Generate zero-shot prompt variants for dog behavior recognition.\n"
        f"Labels: {list(labels)}\n"
        f"Camera context: {camera_context}\n"
        f"Variants per label: {n_variants}\n"
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
        prompts = [str(p).strip() for p in item.get("prompts", []) if str(p).strip()]
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
