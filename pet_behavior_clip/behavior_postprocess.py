"""Post-processing helpers for frame-wise behavior labels and segments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import pandas as pd

_METADATA_COLS = {"timestamp", "anomaly_score", "is_anomaly"}
_ANOMALY_LABEL = "anomaly"
_UNCERTAIN_LABEL = "uncertain"


def infer_frame_behaviors(
    detected: pd.DataFrame,
    confidence_threshold: float = 0.45,
    anomaly_label: str = _ANOMALY_LABEL,
    uncertain_label: str = _UNCERTAIN_LABEL,
    score_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Infer a behavior label for each frame using max score with anomaly-first rule."""
    if detected.empty:
        result = detected.copy()
        result["behavior_raw"] = pd.Series(dtype="object")
        result["behavior_score"] = pd.Series(dtype=float)
        result["behavior_confident"] = pd.Series(dtype=bool)
        result["behavior_label"] = pd.Series(dtype="object")
        return result

    cols = list(score_cols) if score_cols is not None else _resolve_score_cols(detected)
    if not cols:
        raise ValueError("No behavior score columns found for behavior inference.")

    score_df = detected[cols]
    max_scores = score_df.max(axis=1)
    max_labels = score_df.idxmax(axis=1).map(_pretty_label)

    result = detected.copy()
    result["behavior_raw"] = max_labels
    result["behavior_score"] = max_scores
    result["behavior_confident"] = max_scores >= confidence_threshold

    labels: List[str] = []
    anomaly_flags = result["is_anomaly"] if "is_anomaly" in result.columns else [False] * len(result)
    for is_anomaly, raw, confident in zip(anomaly_flags, max_labels, result["behavior_confident"]):
        if bool(is_anomaly):
            labels.append(anomaly_label)
        elif bool(confident):
            labels.append(raw)
        else:
            labels.append(uncertain_label)
    result["behavior_label"] = labels
    return result


def smooth_behavior_labels(
    labeled: pd.DataFrame,
    window_seconds: float = 2.0,
    label_col: str = "behavior_label",
    timestamp_col: str = "timestamp",
    fixed_labels: Iterable[str] = (_ANOMALY_LABEL,),
    output_col: str = "behavior_smooth",
) -> pd.DataFrame:
    """Apply majority voting in a temporal window to reduce label flicker."""
    if labeled.empty:
        result = labeled.copy()
        result[output_col] = pd.Series(dtype="object")
        return result

    if window_seconds <= 0:
        result = labeled.copy()
        result[output_col] = result[label_col]
        return result

    fixed = set(fixed_labels)
    ts = labeled[timestamp_col].to_numpy(dtype=float)
    src = labeled[label_col].astype(str).to_list()

    smoothed: List[str] = []
    for i, current in enumerate(src):
        if current in fixed:
            smoothed.append(current)
            continue

        in_window = (ts >= ts[i] - window_seconds) & (ts <= ts[i] + window_seconds)
        votes = [src[j] for j, ok in enumerate(in_window) if ok and src[j] not in fixed]
        if not votes:
            smoothed.append(current)
            continue

        counts = Counter(votes)
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        smoothed.append(current if current in winners else sorted(winners)[0])

    result = labeled.copy()
    result[output_col] = smoothed
    return result


def build_behavior_segments(
    labeled: pd.DataFrame,
    label_col: str = "behavior_smooth",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Merge contiguous equal labels into behavior segments."""
    if labeled.empty:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s", "label", "frames"])

    required = {label_col, timestamp_col}
    missing = required - set(labeled.columns)
    if missing:
        raise ValueError(f"Missing required columns for segmenting: {sorted(missing)}")

    ordered = labeled[[timestamp_col, label_col]].copy().reset_index(drop=True)
    segments = []

    start_idx = 0
    for i in range(1, len(ordered)):
        if ordered.loc[i, label_col] != ordered.loc[i - 1, label_col]:
            segments.append(_segment_from_slice(ordered, start_idx, i - 1, label_col, timestamp_col))
            start_idx = i
    segments.append(_segment_from_slice(ordered, start_idx, len(ordered) - 1, label_col, timestamp_col))

    return pd.DataFrame(segments)


def summarize_behavior_results(
    labeled: pd.DataFrame,
    segments: pd.DataFrame,
    anomaly_alert_threshold: float = 2.0,
    label_col: str = "behavior_smooth",
) -> str:
    """Render a markdown summary for UI display."""
    total = len(labeled)
    if total == 0:
        return "## Behavior Classification\n\nNo data to display."

    counts = labeled[label_col].value_counts(dropna=False)
    alert_frames = 0
    if "anomaly_score" in labeled.columns:
        alert_frames = int((labeled["anomaly_score"] > anomaly_alert_threshold).sum())

    lines = [
        "## Behavior Classification",
        "",
        f"- Total points: `{total}`",
        f"- Behavior segments: `{len(segments)}`",
        f"- Anomaly alerts (`anomaly_score > {anomaly_alert_threshold:.2f}`): `{alert_frames}` points",
        "",
        "### Label Distribution",
        "",
    ]

    for label, count in counts.items():
        ratio = count / total
        lines.append(f"- `{label}`: {count} ({ratio:.1%})")

    lines += ["", "### Behavior Timeline"]
    for _, row in segments.iterrows():
        lines.append(
            f"- `[{row['start_s']:.2f}-{row['end_s']:.2f} s]` -> {row['label']} "
            f"(duration={row['duration_s']:.2f}s, frames={int(row['frames'])})"
        )

    return "\n".join(lines)


def _resolve_score_cols(detected: pd.DataFrame) -> List[str]:
    return [col for col in detected.columns if col not in _METADATA_COLS]


def _pretty_label(raw_label: str) -> str:
    text = str(raw_label).strip()
    if text.lower().startswith("a picture of an animal "):
        return text[len("a picture of an animal ") :]
    return text


def _segment_from_slice(
    ordered: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    label_col: str,
    timestamp_col: str,
) -> dict:
    start = float(ordered.loc[start_idx, timestamp_col])
    end = float(ordered.loc[end_idx, timestamp_col])
    return {
        "start_s": start,
        "end_s": end,
        "duration_s": max(0.0, end - start),
        "label": str(ordered.loc[start_idx, label_col]),
        "frames": int(end_idx - start_idx + 1),
    }
