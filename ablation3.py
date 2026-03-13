"""ablation3.py - Ablation 3: 4-way Prompt Configuration Human Accuracy

Corresponds to Section 5.3 of the paper.

2 × 2 factorial design
=======================
               max aggregation     mean aggregation
  Single prompt   Config A             Config C
  Multi  prompt   Config B (system)    Config D

Config A  Single + max  : D1 variant only, max agg      (6 prompts)
Config B  Multi  + max  : 5 variants, max agg           (30 prompts) ← current
Config C  Single + mean : D1 variant only, mean agg     (6 prompts, mean = max here)
Config D  Multi  + mean : 5 variants, mean agg          (30 prompts)

Single-pass efficiency
======================
Raw 30-prompt scores are computed once.
  A = aggregate(raw, single_pm,   reducer="max")
  B = aggregate(raw, full_pm,     reducer="max")
  C = aggregate(raw, single_pm,   reducer="max")   # identical to A; included for symmetry
  D = aggregate(raw, full_pm,     reducer="mean")

Plus a 5th baseline condition:
  Config E  Baseline : plain "a dog [label]" prompt, max  (6 prompts, separate pass)

Human ground truth
==================
Loads manual_labels.csv from a prior manual_label.py run.  The frame_idx
and human_label columns are used as ground truth.  Inference is re-run on
the same frame indices so all 5 configs receive identical inputs.

Metrics (per config, per class)
================================
  accuracy   = correct / total
  support    = number of human-labeled frames per class
  macro acc  = mean per-class accuracy (unweighted)
  dynamic    = run + walk group accuracy
  static     = sit + lie + stand group accuracy
  eating     = eating group accuracy

Outputs
=======
  ablation3_accuracy.png   grouped bar chart (5 configs × 6 classes)
  ablation3_results.csv    per-frame predictions for all 5 configs
  ablation3_table.tex      LaTeX table ready for copy-paste

Usage
=====
    python ablation3.py --video dog.mp4 --labels ablation_output/manual_labels.csv
    python ablation3.py --video dog.mp4 --labels ablation_output/manual_labels.csv \\
        --fps 2.0 --out ablation_output/
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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
LABELS  = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL   = "google/siglip-so400m-patch14-224"
FPS     = 2.0

DYNAMIC = ["running", "walking"]
STATIC  = ["sitting", "lying", "standing"]
EATING  = ["eating"]

CONFIG_NAMES = {
    "A": "Single+max",
    "B": "Multi+max",
    "C": "Multi+mean",
    "D": "Baseline",
}
CONFIG_COLORS = {
    "A": "#95a5a6",   # grey
    "B": "#3498db",   # blue   ← current system
    "C": "#2ecc71",   # green
    "D": "#e67e22",   # orange
}


# ── video / model ─────────────────────────────────────────────────────────────
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


# ── inference ──────────────────────────────────────────────────────────────────
def build_baseline_pm() -> dict:
    """One plain prompt per class: 'a dog [label]'."""
    return {lbl: [f"a dog {lbl}"] for lbl in LABELS}


def run_all_configs(clf, frames, timestamps):
    """Single 30-prompt pass + one 6-prompt baseline pass.

    Returns a dict:
        "raw_multi"   : DataFrame of 30 raw prompt scores
        "raw_base"    : DataFrame of 6 baseline prompt scores
        "full_pm"     : full prompt map (5 variants × 6 classes)
        "single_pm"   : D1 variant only
        "baseline_pm" : plain 'a dog [label]' prompts
    """
    res      = build_label_prompt_result(labels=LABELS, mode="template")
    full_pm  = res["prompt_map"]
    single_pm = {lbl: [variants[0]] for lbl, variants in full_pm.items()}
    base_pm  = build_baseline_pm()

    print("[pass 1/2] Multi-prompt (30 prompts) ...")
    raw_multi = clf.classify_frames(frames, flatten_prompt_map(full_pm), timestamps)

    print("[pass 2/2] Baseline (6 plain prompts) ...")
    raw_base  = clf.classify_frames(frames, flatten_prompt_map(base_pm), timestamps)

    return {
        "raw_multi":   raw_multi,
        "raw_base":    raw_base,
        "full_pm":     full_pm,
        "single_pm":   single_pm,
        "baseline_pm": base_pm,
    }


def derive_config_scores(inference: dict) -> dict[str, pd.DataFrame]:
    """Derive per-config aggregated scores from cached raw passes."""
    raw_m = inference["raw_multi"]
    raw_b = inference["raw_base"]
    pm    = inference["full_pm"]
    spm   = inference["single_pm"]
    bpm   = inference["baseline_pm"]

    return {
        "A": aggregate_prompt_scores(raw_m, spm,  reducer="max"),   # Single + max
        "B": aggregate_prompt_scores(raw_m, pm,   reducer="max"),   # Multi  + max
        "C": aggregate_prompt_scores(raw_m, pm,   reducer="mean"),  # Multi  + mean
        "D": aggregate_prompt_scores(raw_b, bpm,  reducer="max"),   # Baseline
    }


# ── evaluation ────────────────────────────────────────────────────────────────
def top_pred(row: pd.Series) -> str:
    return max(LABELS, key=lambda l: float(row[l]))


def evaluate(human_labels: list[str], frame_indices: list[int],
             scores: dict[str, pd.DataFrame]) -> dict:
    """Compute per-class and overall accuracy for each config.

    Returns nested dict:
        result[cfg][label] = {"correct": int, "total": int}
        result[cfg]["macro_acc"] = float
    """
    result: dict[str, dict] = {cfg: defaultdict(lambda: [0, 0]) for cfg in scores}

    for human, idx in zip(human_labels, frame_indices):
        for cfg, df in scores.items():
            pred = top_pred(df.iloc[idx])
            result[cfg][human][1] += 1
            if pred == human:
                result[cfg][human][0] += 1

    # Compute macro accuracy per config
    for cfg in result:
        cls_accs = []
        for lbl in LABELS:
            ok, tot = result[cfg][lbl]
            if tot > 0:
                cls_accs.append(ok / tot)
        result[cfg]["macro_acc"] = float(np.mean(cls_accs)) if cls_accs else 0.0

    return result


def group_acc(result_cfg: dict, group: list[str]) -> tuple[int, int]:
    ok  = sum(result_cfg[l][0] for l in group if result_cfg[l][1] > 0)
    tot = sum(result_cfg[l][1] for l in group if result_cfg[l][1] > 0)
    return ok, tot


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(result: dict, out_path: Path, n_labeled: int) -> None:
    configs = list(CONFIG_NAMES.keys())
    n_labels = len(LABELS)
    x = np.arange(n_labels)
    width = 0.18
    offsets = np.linspace(-(len(configs) - 1) / 2, (len(configs) - 1) / 2, len(configs)) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, cfg in enumerate(configs):
        accs = []
        for lbl in LABELS:
            ok, tot = result[cfg][lbl]
            accs.append(ok / tot * 100 if tot > 0 else 0.0)
        bars = ax.bar(
            x + offsets[i], accs, width * 0.92,
            label=f"{CONFIG_NAMES[cfg]}",
            color=CONFIG_COLORS[cfg],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{acc:.0f}",
                    ha="center", va="bottom", fontsize=7,
                )

    # Macro accuracy horizontal markers
    for cfg in configs:
        macro = result[cfg]["macro_acc"] * 100
        ax.axhline(
            macro, color=CONFIG_COLORS[cfg],
            linestyle="--", linewidth=1.0, alpha=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in LABELS], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Ablation 3 — 4-Configuration Prompt Accuracy  (n={n_labeled} human-labeled frames)\n"
        "Dashed lines = macro accuracy.  "
        "A: Single+max  B: Multi+max (current)  C: Multi+mean  D: Baseline",
        fontsize=9, loc="left",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ── LaTeX table ───────────────────────────────────────────────────────────────
def make_latex_table(result: dict, n_labeled: int) -> str:
    configs = list(CONFIG_NAMES.keys())

    header_cols = " & ".join(
        f"\\textbf{{{CONFIG_NAMES[cfg]}}}" for cfg in configs
    )
    lines = [
        "% Ablation 3 — 4-way prompt configuration accuracy",
        f"% n = {n_labeled} human-labeled frames",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(configs) + "}",
        "\\toprule",
        f"\\textbf{{Class}} & {header_cols} \\\\",
        "\\midrule",
    ]

    def fmt(ok, tot):
        if tot == 0:
            return "---"
        return f"{ok/tot*100:.0f}\\% ({ok}/{tot})"

    for lbl in LABELS:
        cells = " & ".join(fmt(*result[cfg][lbl]) for cfg in configs)
        lines.append(f"{lbl.capitalize()} & {cells} \\\\")

    lines.append("\\midrule")

    # Group rows
    for grp_name, grp in [("Dynamic (run/walk)", DYNAMIC),
                           ("Static (sit/lie/std)", STATIC),
                           ("Eating", EATING)]:
        cells = []
        for cfg in configs:
            ok, tot = group_acc(result[cfg], grp)
            cells.append(fmt(ok, tot))
        lines.append(f"\\textit{{{grp_name}}} & {' & '.join(cells)} \\\\")

    lines.append("\\midrule")
    macro_cells = " & ".join(
        f"{result[cfg]['macro_acc']*100:.1f}\\%" for cfg in configs
    )
    lines += [
        f"\\textbf{{Macro acc.}} & {macro_cells} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Ablation 3: Per-class accuracy under four prompt configurations. "
        "Config B (Multi+max) is the current production system. "
        f"Results are based on {n_labeled} manually labeled frames.}}",
        "\\label{tab:ablation3}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── console report ────────────────────────────────────────────────────────────
def print_report(result: dict, n_labeled: int) -> None:
    configs = list(CONFIG_NAMES.keys())
    col_w = 14

    header = f"  {'Class':<12}" + "".join(f"{CONFIG_NAMES[c]:>{col_w}}" for c in configs)
    print("\n" + "=" * (12 + col_w * len(configs) + 4))
    print("Ablation 3 — 4-way Prompt Configuration Accuracy")
    print(f"n = {n_labeled} human-labeled frames")
    print("=" * (12 + col_w * len(configs) + 4))
    print(header)
    print("  " + "-" * (10 + col_w * len(configs)))

    for lbl in LABELS:
        row = f"  {lbl.capitalize():<12}"
        for cfg in configs:
            ok, tot = result[cfg][lbl]
            row += f"{ok/tot*100:>{col_w-3}.0f}% ({ok}/{tot})" if tot else f"{'---':>{col_w}}"
        print(row)

    print("  " + "-" * (10 + col_w * len(configs)))
    for grp_name, grp in [("Dyn(run/walk)", DYNAMIC),
                           ("Sta(sit/lie/std)", STATIC),
                           ("Eating", EATING)]:
        row = f"  {grp_name:<12}"
        for cfg in configs:
            ok, tot = group_acc(result[cfg], grp)
            row += f"{ok/tot*100:>{col_w-3}.0f}% ({ok}/{tot})" if tot else f"{'---':>{col_w}}"
        print(row)

    print("  " + "-" * (10 + col_w * len(configs)))
    row = f"  {'Macro acc.':<12}"
    for cfg in configs:
        row += f"{result[cfg]['macro_acc']*100:>{col_w-1}.1f}%"
    print(row)
    print("=" * (12 + col_w * len(configs) + 4))

    best = max(configs, key=lambda c: result[c]["macro_acc"])
    print(f"\n  Best macro accuracy: Config {best} ({CONFIG_NAMES[best]}) = "
          f"{result[best]['macro_acc']*100:.1f}%")


# ── CSV ───────────────────────────────────────────────────────────────────────
def save_results_csv(
    human_labels: list[str],
    frame_indices: list[int],
    timestamps: list[float],
    scores: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    rows = []
    for human, idx in zip(human_labels, frame_indices):
        row: dict = {
            "frame_idx":   idx,
            "timestamp_s": round(timestamps[idx], 3),
            "human_label": human,
        }
        for cfg, df in scores.items():
            pred = top_pred(df.iloc[idx])
            conf = round(float(df.iloc[idx][pred]), 4)
            row[f"cfg{cfg}_pred"]  = pred
            row[f"cfg{cfg}_conf"]  = conf
            row[f"cfg{cfg}_agree"] = int(pred == human)
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation 3: 4-way prompt configuration accuracy "
            "(Single/Multi × max/mean + Baseline)"
        )
    )
    parser.add_argument("--video",  required=True,
                        help="Input video path")
    parser.add_argument("--labels", required=True,
                        help="manual_labels.csv from manual_label.py")
    parser.add_argument("--fps",    type=float, default=FPS,
                        help=f"Frame sampling rate (default: {FPS})")
    parser.add_argument("--out",    default="ablation_output",
                        help="Output directory (default: ablation_output/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load human labels ────────────────────────────────────────────────────
    labels_df = pd.read_csv(args.labels)
    required  = {"frame_idx", "human_label"}
    if not required.issubset(labels_df.columns):
        parser.error(f"--labels CSV must contain columns: {required}")

    # Drop any "skip" or missing rows
    labels_df = labels_df[labels_df["human_label"].isin(LABELS)].reset_index(drop=True)
    frame_indices = labels_df["frame_idx"].tolist()
    human_labels  = labels_df["human_label"].tolist()
    n_labeled = len(human_labels)
    print(f"[labels] {n_labeled} annotated frames loaded from {args.labels}")

    # ── load video ───────────────────────────────────────────────────────────
    print("\n=== Loading video ===")
    frames, timestamps = load_video(args.video, args.fps)

    max_idx = max(frame_indices)
    if max_idx >= len(frames):
        parser.error(
            f"frame_idx {max_idx} in labels CSV exceeds video frame count "
            f"({len(frames)}).  Re-run manual_label.py with --fps {args.fps}."
        )

    # ── load model ───────────────────────────────────────────────────────────
    print("\n=== Loading SigLIP model ===")
    clf = load_model()

    # ── inference (two passes total) ─────────────────────────────────────────
    print("\n=== Running inference ===")
    inference = run_all_configs(clf, frames, timestamps)
    scores    = derive_config_scores(inference)

    # ── evaluate ─────────────────────────────────────────────────────────────
    print("\n=== Evaluating ===")
    result = evaluate(human_labels, frame_indices, scores)

    # ── report ───────────────────────────────────────────────────────────────
    print_report(result, n_labeled)

    # ── save outputs ─────────────────────────────────────────────────────────
    print("\n=== Saving outputs ===")

    plot_results(result, out_dir / "ablation3_accuracy.png", n_labeled)

    save_results_csv(
        human_labels, frame_indices, timestamps, scores,
        out_dir / "ablation3_results.csv",
    )

    latex = make_latex_table(result, n_labeled)
    tex_path = out_dir / "ablation3_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"[saved] {tex_path}")

    print("\nDone.")
    print(f"\nTo include in paper, copy: {tex_path}")


if __name__ == "__main__":
    main()
