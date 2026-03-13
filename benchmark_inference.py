"""benchmark_inference.py — Inference speed comparison

Measures wall-clock time and per-frame latency for three inference strategies:

  Strategy A: Baseline
    — No text cache: re-encodes all 30 prompts on every classify_frames call
    — batch_size=1 (single-frame at a time)

  Strategy B: Text Cache
    — Text embeddings pre-computed once and reused across frames
    — batch_size=1 (single-frame at a time)

  Strategy C: Text Cache + Batch Inference
    — Text embeddings pre-computed once
    — All frames processed in a single batched forward pass (batch_size=N)

Usage:
    python benchmark_inference.py --video dog.mp4
    python benchmark_inference.py --video dog.mp4 --fps 2.0 --batch_size 8 --repeats 3

Output:
    benchmark_output/benchmark_results.png
    benchmark_output/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent if _here.name == "pet_behavior_clip" else str(_here)))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)

LABELS     = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL_NAME = "google/siglip-so400m-patch14-224"


# ── model + prompt setup ──────────────────────────────────────────────────────
def load_classifier_and_prompts():
    clf = SigLIPClassifier(model_name=MODEL_NAME, batch_size=1)
    clf._load()

    res     = build_label_prompt_result(labels=LABELS, mode="template")
    pm      = res["prompt_map"]
    prompts = flatten_prompt_map(pm)          # 30 prompts
    device  = clf._device
    print(f"[model] device={device}  prompts={len(prompts)}")
    return clf, pm, prompts, device


# ── Strategy A: no text cache, batch_size=1 ───────────────────────────────────
def strategy_a_no_cache(clf, frames, pm, prompts) -> float:
    """Simulate no-cache: create a fresh classifier clone per frame to force
    text re-encoding each time (mirrors the cost of re-encoding every call)."""
    t0 = time.perf_counter()
    for frame in frames:
        # Re-encode text every frame by calling _encode_text directly
        _ = clf._encode_text(prompts)
        # Image encode (single frame)
        raw = clf.classify_frames([frame], prompts, [0.0])
        _ = aggregate_prompt_scores(raw, pm, reducer="max")
    return time.perf_counter() - t0


# ── Strategy B: text cache, batch_size=1 ─────────────────────────────────────
def strategy_b_cache_single(clf, frames, pm, prompts) -> float:
    """Pre-compute text embeddings once, then process one frame at a time."""
    # Pre-cache: encode text once
    text_feats = clf._encode_text(prompts)

    t0 = time.perf_counter()
    for frame in frames:
        # Bypass text encode — use cached features directly
        img_inputs = clf._processor(
            text=None, images=[frame], return_tensors="pt", padding=True
        )
        img_inputs = {
            k: v.to(clf._device)
            for k, v in img_inputs.items()
            if k == "pixel_values"
        }
        with torch.inference_mode():
            img_feats = clf._model.get_image_features(**img_inputs)
            if hasattr(img_feats, "pooler_output"):
                img_feats = img_feats.pooler_output
            logit_scale = clf._model.logit_scale.exp()
            if hasattr(clf._model, "logit_bias"):
                logits = (img_feats @ text_feats.T) * logit_scale + clf._model.logit_bias
            else:
                logits = (img_feats @ text_feats.T) * logit_scale
            _ = logits.softmax(dim=-1).cpu().numpy()
    return time.perf_counter() - t0


# ── Strategy C: text cache + batch inference ──────────────────────────────────
def strategy_c_cache_batch(clf, frames, pm, prompts, batch_size: int) -> float:
    """Pre-compute text embeddings once, process all frames in batches."""
    text_feats = clf._encode_text(prompts)

    t0 = time.perf_counter()
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        img_inputs = clf._processor(
            text=None, images=chunk, return_tensors="pt", padding=True
        )
        img_inputs = {
            k: v.to(clf._device)
            for k, v in img_inputs.items()
            if k == "pixel_values"
        }
        with torch.inference_mode():
            img_feats = clf._model.get_image_features(**img_inputs)
            if hasattr(img_feats, "pooler_output"):
                img_feats = img_feats.pooler_output
            logit_scale = clf._model.logit_scale.exp()
            if hasattr(clf._model, "logit_bias"):
                logits = (img_feats @ text_feats.T) * logit_scale + clf._model.logit_bias
            else:
                logits = (img_feats @ text_feats.T) * logit_scale
            _ = logits.softmax(dim=-1).cpu().numpy()
    return time.perf_counter() - t0


# ── benchmark runner ──────────────────────────────────────────────────────────
def run_benchmark(clf, frames, pm, prompts, batch_size, repeats) -> dict:
    n = len(frames)
    print(f"\n[benchmark] {n} frames  batch_size={batch_size}  repeats={repeats}")

    results = {"n_frames": n, "batch_size": batch_size}

    strategies = [
        ("A_no_cache",     strategy_a_no_cache,     {}),
        ("B_cache_single", strategy_b_cache_single, {}),
        ("C_cache_batch",  strategy_c_cache_batch,  {"batch_size": batch_size}),
    ]

    for name, fn, kwargs in strategies:
        times = []
        for r in range(repeats):
            t = fn(clf, frames, pm, prompts, **kwargs)
            times.append(t)
            print(f"  [{name}] run {r+1}/{repeats}: {t:.3f}s  "
                  f"({t/n*1000:.1f} ms/frame)")
        mean_t = float(np.mean(times))
        results[name] = {
            "total_s":      round(mean_t, 3),
            "ms_per_frame": round(mean_t / n * 1000, 2),
            "speedup_vs_A": None,
        }

    base = results["A_no_cache"]["ms_per_frame"]
    for k in ["A_no_cache", "B_cache_single", "C_cache_batch"]:
        results[k]["speedup_vs_A"] = round(base / results[k]["ms_per_frame"], 2)

    return results


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: dict, out_path: Path) -> None:
    labels  = [
        "A\nNo Cache\nSingle",
        "B\nText Cache\nSingle",
        f"C\nText Cache\nBatch(×{results['batch_size']})",
    ]
    keys    = ["A_no_cache", "B_cache_single", "C_cache_batch"]
    ms      = [results[k]["ms_per_frame"]  for k in keys]
    speedup = [results[k]["speedup_vs_A"]  for k in keys]
    colors  = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    bars = ax1.bar(labels, ms, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ms)*0.02,
                 f"{val:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Latency per Frame (ms)", fontsize=10)
    ax1.set_title("Inference Latency (lower is better)", fontsize=10)
    ax1.set_ylim(0, max(ms) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(labels, speedup, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars2, speedup):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"×{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Speedup (vs Strategy A)", fontsize=10)
    ax2.set_title("Speedup (higher is better)", fontsize=10)
    ax2.set_ylim(0, max(speedup) * 1.3)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Inference Benchmark: Text Cache vs Batch Inference  "
        f"({results['n_frames']} frames, batch={results['batch_size']})",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] → {out_path}")


def save_csv(results: dict, out_path: Path) -> None:
    rows = []
    for k in ["A_no_cache", "B_cache_single", "C_cache_batch"]:
        rows.append({
            "strategy":     k,
            "n_frames":     results["n_frames"],
            "batch_size":   results["batch_size"],
            "total_s":      results[k]["total_s"],
            "ms_per_frame": results[k]["ms_per_frame"],
            "speedup_vs_A": results[k]["speedup_vs_A"],
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv]  → {out_path}")


def print_summary(results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Inference Benchmark  ({results['n_frames']} frames)")
    print(f"{'='*60}")
    print(f"  {'Strategy':<28} {'ms/frame':>10} {'total(s)':>10} {'speedup':>8}")
    print(f"  {'-'*58}")
    for k, label in [
        ("A_no_cache",     "A  No cache, single frame"),
        ("B_cache_single", "B  Text cache, single frame"),
        ("C_cache_batch",  f"C  Text cache, batch ×{results['batch_size']}"),
    ]:
        r = results[k]
        print(f"  {label:<28} {r['ms_per_frame']:>10.1f} "
              f"{r['total_s']:>10.3f}   ×{r['speedup_vs_A']:.2f}")
    print(f"{'='*60}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: no-cache vs text-cache vs batch inference"
    )
    parser.add_argument("--video",      required=True)
    parser.add_argument("--fps",        type=float, default=2.0)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--repeats",    type=int,   default=3)
    parser.add_argument("--out",        default="benchmark_output")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("=== Loading video ===")
    reader     = VideoReader(args.video, sample_fps=args.fps)
    frame_data = reader.sample_frames()
    reader.release()
    frames = [img for _, img in frame_data]
    print(f"  {len(frames)} frames @ {args.fps} fps")

    print("\n=== Loading model ===")
    clf, pm, prompts, device = load_classifier_and_prompts()

    # warm-up
    print("\n=== Warm-up (3 frames) ===")
    strategy_b_cache_single(clf, frames[:3], pm, prompts)

    results = run_benchmark(clf, frames, pm, prompts,
                            batch_size=args.batch_size,
                            repeats=args.repeats)

    print_summary(results)
    plot_results(results, out_dir / "benchmark_results.png")
    save_csv(results,     out_dir / "benchmark_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()