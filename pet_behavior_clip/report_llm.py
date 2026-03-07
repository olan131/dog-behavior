"""report_llm.py – Generate a natural-language analysis report.

Two modes are supported:

LLM mode (``mode="llm"``)
    Calls the OpenAI Chat Completions API (``gpt-4o-mini`` by default) with a
    structured prompt derived from the detection summary and per-label
    statistics.  Requires ``OPENAI_API_KEY`` to be set in the environment.

Template mode (``mode="template"``)
    Produces a deterministic Markdown report from the same data without any
    external API call.  Used as a fallback when the API key is absent.

The report is returned as a Markdown string and, when *output_path* is given,
also saved to disk.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ReportMode = Literal["llm", "template"]

_DEFAULT_LLM_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You are a veterinary AI assistant that analyses pet behaviour from video data. "
    "Given statistical data about frame-level behaviour confidence scores and detected "
    "anomalies, write a clear, concise report (in Traditional Chinese: 繁體中文) suitable "
    "for a pet owner or veterinarian. Include: (1) overall behaviour summary, "
    "(2) specific anomalies and their timestamps, (3) recommended follow-up actions."
)


def generate_report(
    detected: pd.DataFrame,
    behavior_labels: List[str],
    video_path: Optional[str] = None,
    mode: ReportMode = "template",
    output_path: Optional[str | Path] = None,
    llm_model: str = _DEFAULT_LLM_MODEL,
) -> str:
    """Generate a behaviour analysis report.

    Parameters
    ----------
    detected:
        DataFrame returned by :meth:`~pet_behavior_clip.anomaly.AnomalyDetector.detect`.
        Must contain ``is_anomaly`` and ``anomaly_score`` columns.
    behavior_labels:
        List of behaviour labels that were used during classification.
    video_path:
        Optional path to the source video (used in report metadata).
    mode:
        ``"llm"`` to call OpenAI, ``"template"`` for local generation.
        When ``mode="llm"`` but the API key is absent, the function
        automatically falls back to ``"template"``.
    output_path:
        If provided, the Markdown report is saved to this path.
    llm_model:
        OpenAI model identifier (only used when ``mode="llm"``).

    Returns
    -------
    str
        Markdown-formatted report.
    """
    stats = _compute_stats(detected, behavior_labels)
    summary = _build_summary_dict(detected, stats, video_path)

    if mode == "llm":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set; falling back to template report."
            )
            mode = "template"
        else:
            try:
                report = _llm_report(summary, stats, llm_model, api_key)
            except Exception as exc:
                logger.warning("LLM call failed (%s); using template.", exc)
                mode = "template"

    if mode == "template":
        report = _template_report(summary, stats, behavior_labels)

    if output_path:
        _save_text(report, output_path)
    return report


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _compute_stats(detected: pd.DataFrame, labels: List[str]) -> dict:
    """Compute per-label descriptive statistics."""
    stats: dict = {}
    for label in labels:
        if label not in detected.columns:
            continue
        col = detected[label]
        stats[label] = {
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "max": round(float(col.max()), 4),
            "min": round(float(col.min()), 4),
        }
    return stats


def _build_summary_dict(
    detected: pd.DataFrame, stats: dict, video_path: Optional[str]
) -> dict:
    total = len(detected)
    n_anomaly = int(detected["is_anomaly"].sum()) if "is_anomaly" in detected.columns else 0
    duration = (
        float(detected["timestamp"].iloc[-1])
        if "timestamp" in detected.columns and total > 0
        else None
    )
    worst_ts = None
    if n_anomaly > 0 and "timestamp" in detected.columns and "anomaly_score" in detected.columns:
        worst_ts = float(detected.loc[detected["anomaly_score"].idxmax(), "timestamp"])

    return {
        "video": str(video_path) if video_path else "N/A",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_frames": total,
        "duration_seconds": duration,
        "anomaly_frames": n_anomaly,
        "anomaly_ratio": round(n_anomaly / total, 4) if total > 0 else 0.0,
        "worst_anomaly_timestamp": worst_ts,
    }


# ---------------------------------------------------------------------------
# Template report
# ---------------------------------------------------------------------------


def _template_report(summary: dict, stats: dict, labels: List[str]) -> str:
    lines = [
        "# 寵物行為分析報告",
        "",
        f"**影片來源：** {summary['video']}  ",
        f"**分析時間：** {summary['generated_at']}  ",
        f"**影片長度：** {summary['duration_seconds']:.1f} 秒  " if summary["duration_seconds"] else "",
        f"**分析幀數：** {summary['total_frames']}  ",
        "",
        "---",
        "",
        "## 一、整體行為摘要",
        "",
    ]

    for label, s in stats.items():
        lines.append(
            f"- **{label}**：平均信心度 {s['mean']:.2%}，"
            f"標準差 {s['std']:.2%}，"
            f"最高 {s['max']:.2%}，最低 {s['min']:.2%}"
        )

    lines += [
        "",
        "---",
        "",
        "## 二、異常偵測結果",
        "",
        f"- 異常幀數：**{summary['anomaly_frames']}** / {summary['total_frames']} "
        f"（{summary['anomaly_ratio']:.1%}）",
    ]

    if summary["worst_anomaly_timestamp"] is not None:
        lines.append(
            f"- 最嚴重異常時間點：**{summary['worst_anomaly_timestamp']:.2f} 秒**"
        )

    risk_level = _risk_level(summary["anomaly_ratio"])
    lines += [
        f"- 風險等級：**{risk_level}**",
        "",
        "---",
        "",
        "## 三、建議事項",
        "",
    ]

    if summary["anomaly_ratio"] < 0.05:
        lines.append("寵物行為整體正常，建議繼續保持日常觀察。")
    elif summary["anomaly_ratio"] < 0.20:
        lines.append(
            "偵測到輕度異常行為。建議記錄發生時間並觀察是否持續，"
            "如情況未改善請諮詢獸醫。"
        )
    else:
        lines.append(
            "**偵測到顯著異常行為，建議儘快諮詢獸醫進行評估。**"
        )

    lines += [
        "",
        "---",
        "",
        "## 四、注意事項",
        "",
        "> 本報告由自動化模型生成，僅供參考，不構成醫療診斷。"
        "如有疑慮，請尋求專業獸醫意見。",
        "",
    ]
    return "\n".join(lines)


def _risk_level(ratio: float) -> str:
    if ratio < 0.05:
        return "低風險 🟢"
    if ratio < 0.20:
        return "中風險 🟡"
    return "高風險 🔴"


# ---------------------------------------------------------------------------
# LLM report
# ---------------------------------------------------------------------------


def _llm_report(
    summary: dict, stats: dict, model: str, api_key: str
) -> str:  # pragma: no cover – requires live API
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_content = (
        "以下是寵物行為影片的分析數據，請根據此數據撰寫完整的繁體中文分析報告：\n\n"
        f"**摘要統計：**\n```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**各行為信心度統計：**\n```json\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n```"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("OpenAI API returned an empty response.")
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _save_text(text: str, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("Saved report → %s", out)
