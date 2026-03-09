"""report.py - Local markdown report generation for behavior analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd


def generate_report(
    detected: pd.DataFrame,
    behavior_labels: List[str],
    video_path: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a local deterministic markdown report.

    This function never calls external APIs and is safe for offline execution.
    """
    total = int(len(detected))
    anomaly_frames = int(detected["is_anomaly"].sum()) if "is_anomaly" in detected.columns else 0
    anomaly_ratio = (anomaly_frames / total) if total > 0 else 0.0

    duration = None
    if total > 0 and "timestamp" in detected.columns:
        duration = float(detected["timestamp"].iloc[-1])

    lines = [
        "# Pet Behavior Analysis Report",
        "",
        f"- Video: `{video_path or 'N/A'}`",
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Frames analyzed: `{total}`",
        f"- Duration (s): `{duration:.2f}`" if duration is not None else "- Duration (s): `N/A`",
        f"- Anomaly frames: `{anomaly_frames}` ({anomaly_ratio:.2%})",
        "",
        "## Per-label statistics",
        "",
    ]

    for label in behavior_labels:
        if label not in detected.columns:
            continue
        col = detected[label]
        lines.append(
            f"- `{label}`: mean={float(col.mean()):.4f}, std={float(col.std()):.4f}, "
            f"min={float(col.min()):.4f}, max={float(col.max()):.4f}"
        )

    if anomaly_frames > 0 and {"timestamp", "anomaly_score"}.issubset(detected.columns):
        worst_ts = float(detected.loc[detected["anomaly_score"].idxmax(), "timestamp"])
        lines += ["", "## Peak anomaly", "", f"- Timestamp: `{worst_ts:.2f}` seconds"]

    lines += ["", "## Notes", "", "- This report is generated locally without cloud services."]
    report = "\n".join(lines)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report
