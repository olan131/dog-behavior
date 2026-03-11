"""ablation1.py - Ablation 1: Single Prompt vs Multi-Prompt (template)

Corresponds to Section 5.1 of the paper.

Metric : per-class confidence score standard deviation (grouped bar chart)
Question: Does expanding to 5 multi-viewpoint prompts per class improve
          score discriminability for dynamic classes and reduce noise for
          static classes compared to using a single prompt per class?

Design:
  Single Prompt (off)  — only the FIRST prompt variant per class (6 prompts total)
  Multi-Prompt (template, max) — all 5 variants per class (30 prompts total),
                                  then max aggregation + row normalisation

Expected pattern:
  Dynamic classes (running, walking): Multi std > Single std  → better discriminability
  Static classes  (eating, lying):    Multi std < Single std  → more stable predictions

Usage:
    python ablation1.py --video dog.mp4
    python ablation1.py --video dog.mp4 --fps 2.0 --out results/

Output:
    ablation1_prompt_mode.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- project root detection --
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent if _here.name == "pet_behavior_clip" else _here))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)

# ── constants ────────────────────────────────────────────────────────────────
LABELS = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL  = "google/siglip-so400m-patch14-224"
FPS    = 2.0

_PALETTE = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]


# ── video / model helpers ────────────────────────────────────────────────────
def load_video(path: str, fps: float):
    reader = VideoReader(path, sample_fps=fps)
    data   = reader.sample_frames()
    reader.release()
    print(f"[video] {len(data)} frames @ {fps} fps  ← {path}")
    return [img for _, img in data], [t for t, _ in data]


def load_model() -> SigLIPClassifier:
    clf = SigLIPClassifier(model_name=MODEL)
    clf._load()
    return clf


# ── inference modes ──────────────────────────────────────────────────────────
def run_single_prompt(clf, frames, timestamps) -> pd.DataFrame:
    """One prompt per class (first variant only).  6 prompts total."""
    res       = build_label_prompt_result(labels=LABELS, mode="template")
    single_pm = {lbl: [prompts[0]] for lbl, prompts in res["prompt_map"].items()}
    raw       = clf.classify_frames(frames, flatten_prompt_map(single_pm), timestamps)
    return aggregate_prompt_scores(raw, single_pm, reducer="max")


def run_multi_prompt(clf, frames, timestamps) -> pd.DataFrame:
    """Five prompts per class (template mode), max aggregation.  30 prompts total."""
    res = build_label_prompt_result(labels=LABELS, mode="template")
    pm  = res["prompt_map"]
    raw = clf.classify_frames(frames, flatten_prompt_map(pm), timestamps)
    return aggregate_prompt_scores(raw, pm, reducer="max")


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_std_comparison(
    df_single: pd.DataFrame,
    df_multi:  pd.DataFrame,
    out_path:  Path,
) -> None:
    label_short = [l.capitalize() for l in LABELS]
    std_single  = [df_single[l].std() for l in LABELS]
    std_multi   = [df_multi[l].std()  for l in LABELS]

    x     = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_s = ax.bar(x - width / 2, std_single, width,
                    label="Single Prompt (off)",
                    color="#95a5a6", alpha=0.85, edgecolor="white")
    bars_m = ax.bar(x + width / 2, std_multi,  width,
                    label="Multi-Prompt (template, max)",
                    color=_PALETTE[:len(LABELS)], alpha=0.85, edgecolor="white")

    # value labels on each bar
    for bar in list(bars_s) + list(bars_m):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    # delta annotation above each pair
    for i, (s, m) in enumerate(zip(std_single, std_multi)):
        delta = m - s
        color = "#27ae60" if delta > 0 else "#e74c3c"
        ax.text(x[i], max(s, m) + 0.022, f"Δ{delta:+.3f}",
                ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(label_short, fontsize=10)
    ax.set_ylabel("Confidence Score Std Dev", fontsize=10)
    ax.set_ylim(0, max(max(std_single), max(std_multi)) * 1.45)
    ax.set_title(
        "Ablation 1 — Single Prompt vs Multi-Prompt: Per-class Confidence Std Dev\n"
        "↑ Dynamic classes (running/walking): higher std = better discriminability\n"
        "↓ Static classes  (eating/lying):    lower std  = more stable predictions",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ── console summary ───────────────────────────────────────────────────────────
def print_summary(df_single: pd.DataFrame, df_multi: pd.DataFrame) -> None:
    print(f"\n  {'Label':<12} {'Single':>8} {'Multi':>8} {'Δ std':>8}  "
          f"{'Single med':>11} {'Multi med':>10}")
    print(f"  {'-'*60}")
    for l in LABELS:
        s_std = df_single[l].std()
        m_std = df_multi[l].std()
        s_med = df_single[l].median()
        m_med = df_multi[l].median()
        print(f"  {l.capitalize():<12} {s_std:>8.4f} {m_std:>8.4f} "
              f"{m_std - s_std:>+8.4f}  {s_med:>11.4f} {m_med:>10.4f}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation 1: Single Prompt vs Multi-Prompt std dev comparison"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--fps",   type=float, default=FPS,
                        help=f"Frame sampling rate (default: {FPS})")
    parser.add_argument("--out",   default="ablation_output",
                        help="Output directory (default: ablation_output/)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("=== Loading video ===")
    frames, timestamps = load_video(args.video, args.fps)

    print("\n=== Loading SigLIP model ===")
    clf = load_model()

    print("\n=== Running Single Prompt inference (6 prompts) ===")
    df_single = run_single_prompt(clf, frames, timestamps)

    print("\n=== Running Multi-Prompt inference (30 prompts, max agg) ===")
    df_multi  = run_multi_prompt(clf, frames, timestamps)

    print("\n=== Results ===")
    print_summary(df_single, df_multi)

    print("\n=== Saving figure ===")
    plot_std_comparison(df_single, df_multi, out_dir / "ablation1_prompt_mode.png")

    print("\nDone.")


if __name__ == "__main__":
    main()