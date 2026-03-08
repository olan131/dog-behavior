"""run_ablation.py - Ablation study for pet-behavior-clip.

Three experiments, each targeting one engineering challenge in the paper:

  Ablation 1 (Section 4.2) - Camera context in prompt vs no context
      Metric: per-class confidence score distribution (boxplot)
      Question: Does adding overhead/top-down keywords raise confidence?

  Ablation 2 (Section 4.3) - max vs mean prompt aggregation
      Metric: per-class confidence score distribution (boxplot)
      Question: Does mean aggregation collapse scores due to logit compression?

  Ablation 3 (Section 4.4) - No smoothing vs rolling_mean
      Metric: confidence timeline, raw vs smoothed
      Question: Does temporal smoothing improve readability of behavior trends?

Usage:
    python run_ablation.py --video dog.mp4
    python run_ablation.py --video dog.mp4 --fps 2.0

Outputs (saved to ./ablation_output/):
    ablation1_camera_context.png
    ablation2_aggregation.png
    ablation3_smoothing.png
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

# -- Auto-detect project root (works regardless of script location) --
_here = Path(__file__).resolve().parent
if _here.name == "pet_behavior_clip":
    _root = _here.parent
else:
    _root = _here
sys.path.insert(0, str(_root))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt_llm import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)
from pet_behavior_clip.smoothing import smooth_scores

# -- Constants --
LABELS = [
    "active",
    "resting",
    "eating/drinking",
]
MODEL = "google/siglip-so400m-patch14-224"
FPS   = 2.0
OUT   = Path("ablation_output")

# Camera context strings for Ablation 1
CTX_GENERIC  = "indoor pet camera"                       # no viewpoint info
CTX_OVERHEAD = "overhead top-down surveillance camera"   # explicit viewpoint

_PALETTE = ["#3498db", "#e74c3c", "#2ecc71"]
_LABEL_SHORT = {
    "active":          "Active",
    "resting":         "Resting",
    "eating/drinking": "Eating/Drinking",
}


# ====================================================================
# Step 0 - Load video and initialize classifier (once)
# ====================================================================
def load_video(video_path: str, fps: float):
    reader = VideoReader(video_path, sample_fps=fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames     = [img for _, img in frame_data]
    print(f"[video] {len(frames)} frames @ {fps} fps from {video_path}")
    return frames, timestamps


def get_classifier() -> SigLIPClassifier:
    clf = SigLIPClassifier(model_name=MODEL)
    clf._load()
    return clf


def _run_inference(clf, frames, timestamps, camera_context: str, reducer: str = "max"):
    """Run full pipeline for given camera_context and reducer, return score DataFrame."""
    res = build_label_prompt_result(
        labels=LABELS,
        mode="template",
        camera_context=camera_context,
    )
    pm  = res["prompt_map"]
    pl  = flatten_prompt_map(pm)
    raw = clf.classify_frames(frames, pl, timestamps)
    df  = aggregate_prompt_scores(raw, pm, reducer=reducer)
    return df


def _boxplot_side(ax, df, title, label_short, colors, show_ylabel=True):
    """Draw a single boxplot panel."""
    data = [df[l].to_numpy() for l in LABELS]
    bp = ax.boxplot(data, patch_artist=True, notch=False, tick_labels=label_short)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i + 1, med + 0.02, f"{med:.3f}", ha="center", fontsize=9, color="black")
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("Confidence Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)


# ====================================================================
# Ablation 1 - Camera context: generic vs overhead
#   Corresponds to Section 4.2 (viewpoint domain shift)
#   Metric: per-class confidence boxplot
# ====================================================================
def ablation1_camera_context(clf, frames, timestamps, out_dir: Path):
    print("[ablation 1] generic context vs overhead context ...")

    df_generic  = _run_inference(clf, frames, timestamps, CTX_GENERIC)
    df_overhead = _run_inference(clf, frames, timestamps, CTX_OVERHEAD)

    label_short = [_LABEL_SHORT[l] for l in LABELS]
    colors      = _PALETTE[:len(LABELS)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _boxplot_side(axes[0], df_generic,  f'Generic context\n("{CTX_GENERIC}")',  label_short, colors, show_ylabel=True)
    _boxplot_side(axes[1], df_overhead, f'Overhead context\n("{CTX_OVERHEAD}")', label_short, colors, show_ylabel=False)

    fig.suptitle(
        "Ablation 1: Effect of Camera Context on Confidence Score\n"
        "(Does adding overhead/top-down keywords improve score?)",
        fontsize=11
    )
    fig.tight_layout()

    path = out_dir / "ablation1_camera_context.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

    for label, ctx_name, df in [
        ("generic",  CTX_GENERIC,  df_generic),
        ("overhead", CTX_OVERHEAD, df_overhead),
    ]:
        for l in LABELS:
            s = _LABEL_SHORT[l]
            print(f"  [{ctx_name[:15]}] {s}: median={df[l].median():.4f}  std={df[l].std():.4f}")

    return df_overhead  # use overhead scores downstream


# ====================================================================
# Ablation 2 - max vs mean prompt aggregation
#   Corresponds to Section 4.3 (logit compression)
#   Metric: per-class confidence boxplot
# ====================================================================
def ablation2_aggregation(clf, frames, timestamps, out_dir: Path):
    print("[ablation 2] max vs mean aggregation ...")

    df_max  = _run_inference(clf, frames, timestamps, CTX_OVERHEAD, reducer="max")
    df_mean = _run_inference(clf, frames, timestamps, CTX_OVERHEAD, reducer="mean")

    label_short = [_LABEL_SHORT[l] for l in LABELS]
    colors      = _PALETTE[:len(LABELS)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _boxplot_side(axes[0], df_max,  "reducer = max",  label_short, colors, show_ylabel=True)
    _boxplot_side(axes[1], df_mean, "reducer = mean", label_short, colors, show_ylabel=False)

    fig.suptitle("Ablation 2: Prompt Aggregation Strategy (max vs mean)", fontsize=11)
    fig.tight_layout()

    path = out_dir / "ablation2_aggregation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

    for reducer, df in [("max", df_max), ("mean", df_mean)]:
        for l in LABELS:
            s = _LABEL_SHORT[l]
            print(f"  [{reducer}] {s}: median={df[l].median():.4f}  std={df[l].std():.4f}")

    return df_max  # pass max scores to ablation 3


# ====================================================================
# Ablation 3 - no smoothing vs rolling_mean
#   Corresponds to Section 4.4 (temporal instability)
#   Metric: confidence timeline, raw vs smoothed
# ====================================================================
def ablation3_smoothing(overhead_scores: pd.DataFrame, out_dir: Path):
    print("[ablation 3] no smoothing vs rolling_mean ...")

    smoothed = smooth_scores(overhead_scores, window=5, method="rolling_mean")
    ts       = overhead_scores["timestamp"].to_numpy()

    label_short = [_LABEL_SHORT[l] for l in LABELS]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    titles = ["No smoothing (raw)", "rolling_mean (window=5)"]

    for ax, df, title in zip(axes, [overhead_scores, smoothed], titles):
        for i, (label, short) in enumerate(zip(LABELS, label_short)):
            ax.plot(ts, df[label].to_numpy(),
                    label=short, color=_PALETTE[i], linewidth=1.6)
        ax.set_title(title)
        ax.set_ylabel("Confidence Score")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Ablation 3: Effect of Temporal Smoothing on Score Stability", fontsize=11)
    fig.tight_layout()

    path = out_dir / "ablation3_smoothing.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

    # quantify: count label switches
    def label_switches(df):
        top = df[LABELS].idxmax(axis=1)
        return int((top != top.shift()).sum() - 1)

    raw_sw  = label_switches(overhead_scores)
    smth_sw = label_switches(smoothed)
    print(f"  label switches -- raw: {raw_sw}  smoothed: {smth_sw}")


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Pet-behavior-clip ablation study")
    parser.add_argument("--video", required=True, help="Path to input video (e.g. dog.mp4)")
    parser.add_argument("--fps",   type=float, default=FPS, help=f"Sampling FPS (default: {FPS})")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Loading video & model ===")
    frames, timestamps = load_video(args.video, args.fps)
    clf = get_classifier()

    print("\n=== Ablation 1: Camera Context (generic vs overhead) ===")
    overhead_scores = ablation1_camera_context(clf, frames, timestamps, OUT)

    print("\n=== Ablation 2: Aggregation (max vs mean) ===")
    max_scores = ablation2_aggregation(clf, frames, timestamps, OUT)

    print("\n=== Ablation 3: Smoothing ===")
    ablation3_smoothing(max_scores, OUT)

    print(f"\nAll done. Figures saved to: {OUT.resolve()}/")


if __name__ == "__main__":
    main()