"""ablation2.py - Ablation 2: max vs mean Prompt Aggregation

Corresponds to Section 5.2 of the paper.

Metric : per-class confidence score distribution (side-by-side boxplot)
Question: Does mean aggregation dilute the best-matching prompt's signal,
          collapsing inter-class confidence gaps?

Design:
  max  aggregation — take the highest-scoring prompt per class (30→6)
  mean aggregation — average all 5 prompt scores per class  (30→6)
  Both followed by row normalisation so scores sum to 1.

Expected pattern:
  max  → clear separation between dominant and non-dominant classes
  mean → scores compressed toward uniform distribution (inter-class gap collapses)

Usage:
    python ablation2.py --video dog.mp4
    python ablation2.py --video dog.mp4 --fps 2.0 --out results/

Output:
    ablation2_aggregation.png
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

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent if _here.name == "pet_behavior_clip" else _here))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)

# ── constants ─────────────────────────────────────────────────────────────────
LABELS = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL  = "google/siglip-so400m-patch14-224"
FPS    = 2.0

_PALETTE = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]


# ── helpers ───────────────────────────────────────────────────────────────────
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


def run_inference(clf, frames, timestamps, reducer: str) -> pd.DataFrame:
    """30 prompts (template mode), given reducer, then row-normalise."""
    res = build_label_prompt_result(labels=LABELS, mode="template")
    pm  = res["prompt_map"]
    raw = clf.classify_frames(frames, flatten_prompt_map(pm), timestamps)
    return aggregate_prompt_scores(raw, pm, reducer=reducer)


# ── plotting ──────────────────────────────────────────────────────────────────
def _draw_boxplot(ax, df: pd.DataFrame, title: str, show_ylabel: bool = True) -> None:
    label_short = [l.capitalize() for l in LABELS]
    data = [df[l].to_numpy() for l in LABELS]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        tick_labels=label_short,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, c in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # median value labels
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i + 1, med + 0.025, f"{med:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title(title, fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Confidence Score", fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)


def plot_aggregation_comparison(
    df_max:  pd.DataFrame,
    df_mean: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    _draw_boxplot(axes[0], df_max,  "reducer = max  (Multi-Prompt best-match)",
                  show_ylabel=True)
    _draw_boxplot(axes[1], df_mean, "reducer = mean (Multi-Prompt average)",
                  show_ylabel=False)

    fig.suptitle(
        "Ablation 2 — max vs mean Prompt Aggregation: Per-class Confidence Distribution\n"
        "max preserves the best-matching signal; "
        "mean dilutes it with low-matching variants",
        fontsize=10,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ── console summary ───────────────────────────────────────────────────────────
def print_summary(df_max: pd.DataFrame, df_mean: pd.DataFrame) -> None:
    print(f"\n  {'Label':<12} {'max med':>9} {'mean med':>9} {'Δ median':>10}  "
          f"{'max std':>8} {'mean std':>9}")
    print(f"  {'-'*62}")
    for l in LABELS:
        mx_med  = df_max[l].median()
        mn_med  = df_mean[l].median()
        mx_std  = df_max[l].std()
        mn_std  = df_mean[l].std()
        delta   = mx_med - mn_med
        print(f"  {l.capitalize():<12} {mx_med:>9.4f} {mn_med:>9.4f} "
              f"{delta:>+10.4f}  {mx_std:>8.4f} {mn_std:>9.4f}")

    # gap metric: std of class medians (higher = better inter-class separation)
    gap_max  = float(np.std([df_max[l].median()  for l in LABELS]))
    gap_mean = float(np.std([df_mean[l].median() for l in LABELS]))
    print(f"\n  Inter-class median std:  max={gap_max:.4f}  mean={gap_mean:.4f}  "
          f"(higher = better separation)")
    dilution = (gap_max - gap_mean) / gap_max * 100 if gap_max > 0 else 0
    print(f"  Mean aggregation dilution: {dilution:.1f}% gap loss vs max")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation 2: max vs mean prompt aggregation boxplot"
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

    print("\n=== Running max aggregation inference ===")
    df_max  = run_inference(clf, frames, timestamps, reducer="max")

    print("\n=== Running mean aggregation inference ===")
    df_mean = run_inference(clf, frames, timestamps, reducer="mean")

    print("\n=== Results ===")
    print_summary(df_max, df_mean)

    print("\n=== Saving figure ===")
    plot_aggregation_comparison(df_max, df_mean, out_dir / "ablation2_aggregation.png")

    print("\nDone.")


if __name__ == "__main__":
    main()