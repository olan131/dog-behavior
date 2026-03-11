"""plots.py – Visualization utilities for pet-behaviour analysis results.

Provides three main chart types:

* :func:`plot_behavior_timeline` – per-label probability curves over time with
  anomaly markers.
* :func:`plot_anomaly_heatmap` – heatmap of scores across frames × labels,
  highlighting anomalous frames.
* :func:`plot_confidence_distribution` – violin/box plots showing the
  confidence distribution for each behaviour label.

All functions accept an optional *output_path* argument; when supplied the
figure is saved to disk.  The matplotlib ``Figure`` object is always returned
so callers can embed it in a UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")  # non-interactive backend – safe for servers / CI

logger = logging.getLogger(__name__)

_ANOMALY_COLOR = "#e74c3c"
_PALETTE = [
    "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
    "#1abc9c", "#f39c12", "#34495e", "#e91e63",
]


def plot_behavior_timeline(
    scores: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Behaviour Score Timeline",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Line chart: label confidence over time.

    Parameters
    ----------
    scores:
        DataFrame with a ``timestamp`` column and one numeric column per
        behaviour label.  May optionally include ``is_anomaly`` and
        ``anomaly_score`` columns (added by :class:`~pet_behavior_clip.anomaly.AnomalyDetector`).
    output_path:
        If provided, the figure is saved to this path.
    title:
        Figure title.
    figsize:
        Matplotlib figure size in inches ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    label_cols = _label_columns(scores)
    ts = scores["timestamp"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    for i, col in enumerate(label_cols):
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(ts, scores[col].to_numpy(), label=col, color=color, linewidth=1.8)

    # Shade anomaly regions if available
    if "is_anomaly" in scores.columns:
        anomaly_mask = scores["is_anomaly"].to_numpy(dtype=bool)
        _shade_anomaly_regions(ax, ts, anomaly_mask)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


def plot_anomaly_heatmap(
    scores: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Behaviour Score Heatmap",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Heatmap: label × frame grid coloured by confidence.

    Anomalous frames (when ``is_anomaly`` is present) are marked with a red
    border on the right-hand axis.

    Parameters
    ----------
    scores:
        Same DataFrame as :func:`plot_behavior_timeline`.
    output_path:
        Optional save path.
    title:
        Figure title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    label_cols = _label_columns(scores)
    data = scores[label_cols].to_numpy(dtype=float).T  # (L, N)
    ts = scores["timestamp"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        extent=[ts[0], ts[-1], len(label_cols) - 0.5, -0.5],
    )

    ax.set_yticks(range(len(label_cols)))
    ax.set_yticklabels(label_cols, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Confidence")

    # Overlay anomaly markers on x-axis
    if "is_anomaly" in scores.columns:
        anomaly_ts = ts[scores["is_anomaly"].to_numpy(dtype=bool)]
        for at in anomaly_ts:
            ax.axvline(at, color=_ANOMALY_COLOR, alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_confidence_distribution(
    scores: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Confidence Distribution per Behaviour",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Box plots showing confidence distribution for each label.

    Parameters
    ----------
    scores:
        DataFrame with label columns.
    output_path:
        Optional save path.
    title:
        Figure title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    label_cols = _label_columns(scores)
    data = [scores[col].dropna().to_numpy() for col in label_cols]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        orientation="vertical",
        tick_labels=label_cols,
    )

    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


def plot_behavior_segments_timeline(
    segments: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Behavior Timeline Segments",
    figsize: tuple = (12, 2.8),
) -> plt.Figure:
    """Segment chart: contiguous behavior labels over time.

    Parameters
    ----------
    segments:
        DataFrame from ``build_behavior_segments`` with columns ``start_s``,
        ``end_s``, and ``label``.
    output_path:
        Optional save path.
    title:
        Figure title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if segments.empty:
        ax.text(0.5, 0.5, "No behavior segments", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        if output_path:
            _save(fig, output_path)
        return fig

    ordered = segments.sort_values("start_s").reset_index(drop=True)
    start_min = float(ordered["start_s"].min())
    end_max = float(ordered["end_s"].max())
    span = max(1e-6, end_max - start_min)
    min_width = max(0.08, span * 0.005)

    labels = [str(v) for v in ordered["label"].tolist()]
    unique_labels = list(dict.fromkeys(labels))
    color_map = _segment_color_map(unique_labels)

    for _, row in ordered.iterrows():
        start = float(row["start_s"])
        end = float(row["end_s"])
        label = str(row["label"])
        width = max(min_width, end - start)
        ax.barh(
            y=0,
            width=width,
            left=start,
            height=0.56,
            color=color_map[label],
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_xlim(start_min, end_max + min_width)
    ax.grid(axis="x", alpha=0.25)

    handles = [Patch(facecolor=color_map[name], label=name) for name in unique_labels]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=min(6, len(handles)), fontsize=8)

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _label_columns(scores: pd.DataFrame) -> List[str]:
    """Return non-metadata column names."""
    exclude = {"timestamp", "anomaly_score", "is_anomaly"}
    return [c for c in scores.columns if c not in exclude]


def _shade_anomaly_regions(
    ax: plt.Axes,
    timestamps: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Fill vertical spans for consecutive anomaly frames."""
    in_region = False
    start_ts = 0.0
    for i, flag in enumerate(mask):
        if flag and not in_region:
            start_ts = timestamps[i]
            in_region = True
        elif not flag and in_region:
            ax.axvspan(start_ts, timestamps[i - 1], color=_ANOMALY_COLOR, alpha=0.15)
            in_region = False
    if in_region:
        ax.axvspan(start_ts, timestamps[-1], color=_ANOMALY_COLOR, alpha=0.15)


def _save(fig: plt.Figure, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved figure → %s", out)


def _segment_color_map(labels: Sequence[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    palette_i = 0
    for label in labels:
        lowered = label.lower()
        if lowered == "anomaly":
            color_map[label] = _ANOMALY_COLOR
            continue
        if lowered == "uncertain":
            color_map[label] = "#95a5a6"
            continue
        color_map[label] = _PALETTE[palette_i % len(_PALETTE)]
        palette_i += 1
    return color_map
