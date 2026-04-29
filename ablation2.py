"""ablation2.py - Ablation 2: Temporal Smoothing (raw vs rolling_mean)

Corresponds to Section 5.2 of the paper.

Goal
====
Compare per-frame confidence timelines before and after temporal smoothing:
  - Condition A: raw scores (no smoothing)
  - Condition B: rolling_mean smoothing with window=5

Design
======
1) Use the same sampled frames and timestamps for both conditions.
2) Use the same zero-shot classifier and prompt setup (template + max aggregation).
3) Only change one factor: temporal smoothing method.
4) Produce two figures for direct visual comparison:
   - Figure 2a: raw timeline
   - Figure 2b: rolling_mean(window=5) timeline
5) Compute readability-oriented quantitative metrics:
   - Dominant-label switches (lower is more stable)
   - Pairwise curve crossing count (lower is clearer separation)
   - Walking-vs-standing crossing count (target ambiguous pair)
   - Mean top-1 margin (higher is clearer dominance)

Usage
=====
    python ablation2.py --video dog.mp4
    python ablation2.py --video dog.mp4 --fps 2.0 --window 5 --out ablation_output/

Outputs
=======
  ablation2_fig2a_raw.png
  ablation2_fig2b_rolling_mean_w5.png
  ablation2_metrics.csv
  ablation2_pair_crossings.csv
  ablation2_summary.txt
"""

from __future__ import annotations

import argparse
import itertools
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
from pet_behavior_clip.prompt import classify_with_template_max
from pet_behavior_clip.smoothing import smooth_scores

LABELS = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL = "google/siglip-so400m-patch14-224"
FPS = 2.0
WINDOW = 5

_PALETTE = {
    "running": "#1f77b4",
    "eating": "#ff7f0e",
    "walking": "#2ca02c",
    "standing": "#9467bd",
    "sitting": "#d62728",
    "lying": "#17becf",
}


def load_video(path: str, fps: float):
    reader = VideoReader(path, sample_fps=fps)
    data = reader.sample_frames()
    reader.release()
    print(f"[video] {len(data)} frames @ {fps} fps  <- {path}")
    return [img for _, img in data], [t for t, _ in data]


def load_model() -> SigLIPClassifier:
    clf = SigLIPClassifier(model_name=MODEL)
    clf._load()
    return clf


def run_raw_scores(clf: SigLIPClassifier, frames, timestamps) -> pd.DataFrame:
    return classify_with_template_max(
        classifier=clf,
        frames=frames,
        labels=LABELS,
        timestamps=timestamps,
    )


def _dominant_switches(df: pd.DataFrame) -> int:
    seq = df[LABELS].idxmax(axis=1)
    if len(seq) <= 1:
        return 0
    return int((seq != seq.shift(1)).sum() - 1)


def _mean_top1_margin(df: pd.DataFrame) -> float:
    vals = df[LABELS].to_numpy(dtype=float)
    if vals.shape[1] < 2:
        return 0.0
    part = np.partition(vals, -2, axis=1)
    top2 = part[:, -2]
    top1 = part[:, -1]
    return float(np.mean(top1 - top2))


def _crossing_count(a: pd.Series, b: pd.Series) -> int:
    diff = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
    signs = np.sign(diff)

    # Forward-fill zero-sign points to avoid counting tie plateaus as extra crossings.
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(len(signs) - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def pair_crossings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(LABELS, 2):
        rows.append(
            {
                "pair": f"{a}|{b}",
                "crossings": _crossing_count(df[a], df[b]),
            }
        )
    return pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)


def summarize_metrics(raw_df: pd.DataFrame, smooth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_pair = pair_crossings(raw_df)
    sm_pair = pair_crossings(smooth_df)

    merged_pair = raw_pair.merge(sm_pair, on="pair", suffixes=("_raw", "_smooth"))
    merged_pair["delta"] = merged_pair["crossings_smooth"] - merged_pair["crossings_raw"]
    merged_pair["reduction_pct"] = np.where(
        merged_pair["crossings_raw"] > 0,
        (merged_pair["crossings_raw"] - merged_pair["crossings_smooth"]) / merged_pair["crossings_raw"] * 100.0,
        0.0,
    )

    walk_stand = merged_pair.loc[merged_pair["pair"] == "walking|standing"]
    ws_raw = int(walk_stand["crossings_raw"].iloc[0]) if not walk_stand.empty else 0
    ws_smooth = int(walk_stand["crossings_smooth"].iloc[0]) if not walk_stand.empty else 0

    total_raw = int(merged_pair["crossings_raw"].sum())
    total_smooth = int(merged_pair["crossings_smooth"].sum())

    metrics = pd.DataFrame(
        [
            {
                "condition": "raw",
                "dominant_switches": _dominant_switches(raw_df),
                "total_pair_crossings": total_raw,
                "walking_standing_crossings": ws_raw,
                "mean_top1_margin": _mean_top1_margin(raw_df),
            },
            {
                "condition": "rolling_mean_w5",
                "dominant_switches": _dominant_switches(smooth_df),
                "total_pair_crossings": total_smooth,
                "walking_standing_crossings": ws_smooth,
                "mean_top1_margin": _mean_top1_margin(smooth_df),
            },
        ]
    )

    return metrics, merged_pair.sort_values("delta").reset_index(drop=True)


def _plot_timeline(scores: pd.DataFrame, out_path: Path, title: str) -> None:
    ts = scores["timestamp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in LABELS:
        ax.plot(ts, scores[lbl].to_numpy(dtype=float), lw=1.8, label=lbl, color=_PALETTE[lbl])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


def write_summary(metrics: pd.DataFrame, pair_df: pd.DataFrame, out_path: Path) -> None:
    row_raw = metrics.loc[metrics["condition"] == "raw"].iloc[0]
    row_s = metrics.loc[metrics["condition"] == "rolling_mean_w5"].iloc[0]

    def pct_drop(v_raw: float, v_new: float) -> float:
        if v_raw <= 0:
            return 0.0
        return (v_raw - v_new) / v_raw * 100.0

    switch_drop = pct_drop(float(row_raw["dominant_switches"]), float(row_s["dominant_switches"]))
    cross_drop = pct_drop(float(row_raw["total_pair_crossings"]), float(row_s["total_pair_crossings"]))
    ws_drop = pct_drop(float(row_raw["walking_standing_crossings"]), float(row_s["walking_standing_crossings"]))
    margin_gain = float(row_s["mean_top1_margin"]) - float(row_raw["mean_top1_margin"])

    lines = [
        "Ablation 2 Summary: Temporal Smoothing (raw vs rolling_mean window=5)",
        "",
        f"dominant_switches: raw={int(row_raw['dominant_switches'])}, smooth={int(row_s['dominant_switches'])}, drop={switch_drop:.1f}%",
        f"total_pair_crossings: raw={int(row_raw['total_pair_crossings'])}, smooth={int(row_s['total_pair_crossings'])}, drop={cross_drop:.1f}%",
        f"walking_standing_crossings: raw={int(row_raw['walking_standing_crossings'])}, smooth={int(row_s['walking_standing_crossings'])}, drop={ws_drop:.1f}%",
        f"mean_top1_margin: raw={float(row_raw['mean_top1_margin']):.4f}, smooth={float(row_s['mean_top1_margin']):.4f}, gain={margin_gain:+.4f}",
        "",
        "Interpretation:",
        "- Fewer dominant switches and pairwise crossings indicate better timeline readability.",
        "- Fewer walking|standing crossings indicate reduced boundary ambiguity between adjacent classes.",
        "- Larger top-1 margin indicates clearer behavior dominance per frame.",
        "",
        "Top 8 crossing reductions by pair:",
    ]

    top8 = pair_df.sort_values("delta").head(8)
    for _, row in top8.iterrows():
        lines.append(
            f"- {row['pair']}: raw={int(row['crossings_raw'])} -> smooth={int(row['crossings_smooth'])} "
            f"(delta={int(row['delta'])}, reduction={float(row['reduction_pct']):.1f}%)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation 2: Temporal smoothing readability comparison (raw vs rolling_mean)."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--fps", type=float, default=FPS, help=f"Frame sampling rate (default: {FPS})")
    parser.add_argument("--window", type=int, default=WINDOW, help=f"Rolling mean window (default: {WINDOW})")
    parser.add_argument("--out", default="ablation_output", help="Output directory (default: ablation_output/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading video ===")
    frames, timestamps = load_video(args.video, args.fps)

    print("\n=== Loading SigLIP model ===")
    clf = load_model()

    print("\n=== Running raw inference (template + max) ===")
    raw_df = run_raw_scores(clf, frames, timestamps)

    print(f"\n=== Applying rolling_mean smoothing (window={args.window}) ===")
    smooth_df = smooth_scores(raw_df, window=args.window, method="rolling_mean")

    print("\n=== Saving Figure 2a / Figure 2b ===")
    _plot_timeline(
        raw_df,
        out_dir / "ablation2_fig2a_raw.png",
        "Figure 2a. Raw confidence timeline (no smoothing)",
    )
    _plot_timeline(
        smooth_df,
        out_dir / "ablation2_fig2b_rolling_mean_w5.png",
        f"Figure 2b. Smoothed confidence timeline (rolling_mean, window={args.window})",
    )

    print("\n=== Computing metrics ===")
    metrics_df, pair_df = summarize_metrics(raw_df, smooth_df)
    metrics_path = out_dir / "ablation2_metrics.csv"
    pair_path = out_dir / "ablation2_pair_crossings.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    print(f"[csv] saved -> {metrics_path}")
    print(f"[csv] saved -> {pair_path}")

    summary_path = out_dir / "ablation2_summary.txt"
    write_summary(metrics_df, pair_df, summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
