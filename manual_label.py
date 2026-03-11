"""manual_label.py - Human annotation spot-check: Single Prompt vs Multi Prompt

Mirrors the design of ablation1.py (Section 5.1), replacing per-class std
comparison with a human accuracy comparison on the same two conditions.

System A: Single Prompt — 1 prompt per class (D1 variant only), 6 prompts total.
          Hypothesis: dynamic behaviors (running/walking) are hard to distinguish.

System B: Multi Prompt  — 5 prompts per class (template), max aggregation,
          30 prompts total. Current system design.
          Hypothesis: static behaviors (sitting/lying) are harder to distinguish
          due to competition from concrete-posture variants of other classes.

Shared: same SigLIP model, same video, same 6-class label space.
        Only Prompt design differs — isolates the effect of prompt multiplicity.

Usage:
    python manual_label.py --video dog.mp4
    python manual_label.py --video dog.mp4 --n 50 --fps 2.0
    python manual_label.py --video dog.mp4 --multi-reducer mean --sample-mode stratified

Keys:
    1  -> running    4  -> standing
    2  -> eating     5  -> sitting
    3  -> walking    6  -> lying
    s  -> skip
    q  -> quit and save
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent if _here.name == "pet_behavior_clip" else str(_here)))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)

LABELS = ["running", "eating", "walking", "standing", "sitting", "lying"]
MODEL  = "google/siglip-so400m-patch14-224"

KEY_MAP = {
    ord("1"): "running",
    ord("2"): "eating",
    ord("3"): "walking",
    ord("4"): "standing",
    ord("5"): "sitting",
    ord("6"): "lying",
    ord("s"): "skip",
    ord("q"): "quit",
}

WINDOW_TITLE = (
    "1=Running  2=Eating  3=Walking  "
    "4=Standing  5=Sitting  6=Lying  "
    "s=Skip  q=Quit"
)

COL_A_HEAD = (255, 180,  80)   # amber  — System A
COL_B_HEAD = ( 80, 200, 255)   # cyan   — System B
COL_HIT    = ( 60, 220,  60)   # green
COL_MISS   = (110, 110, 110)   # grey
COL_KEY    = (200, 200,  60)   # yellow


# ── inference ─────────────────────────────────────────────────────────────────
def run_systems(clf, frames, timestamps, reducer="max"):
    """Run SysA and SysB in a single forward pass.

    SysA (single prompt) uses only the first variant per class.
    SysB (multi prompt)  uses all 5 variants per class.
    Both share the same 30-prompt raw scores — no double inference.
    """
    print(f"[Inference] Multi Prompt — 5 variants × 6 classes (30 prompts, single pass) ...")
    res       = build_label_prompt_result(labels=LABELS, mode="template")
    pm        = res["prompt_map"]
    single_pm = {lbl: [variants[0]] for lbl, variants in pm.items()}

    raw = clf.classify_frames(frames, flatten_prompt_map(pm), timestamps)

    print(f"[Sys A] Single Prompt — D1 only (aggregating from cached raw scores) ...")
    df_a = aggregate_prompt_scores(raw, single_pm, reducer="max")

    print(f"[Sys B] Multi Prompt  — {reducer} aggregation ...")
    df_b = aggregate_prompt_scores(raw, pm, reducer=reducer)

    return df_a, df_b


def run_system_a(clf, frames, timestamps):
    """Single Prompt: only the first (D1) variant per class, 6 prompts total."""
    print("[Sys A] Single Prompt — D1 variant only (6 prompts) ...")
    res       = build_label_prompt_result(labels=LABELS, mode="template")
    single_pm = {lbl: [variants[0]] for lbl, variants in res["prompt_map"].items()}
    raw       = clf.classify_frames(frames, flatten_prompt_map(single_pm), timestamps)
    return aggregate_prompt_scores(raw, single_pm, reducer="max"), single_pm


def run_system_b(clf, frames, timestamps, reducer="max"):
    """Multi Prompt: all 5 variants per class + aggregation, 30 prompts total."""
    print(f"[Sys B] Multi Prompt — 5 variants + {reducer} (30 prompts) ...")
    res = build_label_prompt_result(labels=LABELS, mode="template")
    pm  = res["prompt_map"]
    raw = clf.classify_frames(frames, flatten_prompt_map(pm), timestamps)
    return aggregate_prompt_scores(raw, pm, reducer=reducer), pm


def top_pred(row, labels):
    return max(labels, key=lambda l: float(row[l]))


def sample_indices(df_a, df_b, n_samples, total, mode):
    n = min(n_samples, total)
    if mode == "random":
        return sorted(random.sample(range(total), n))

    # Pseudo-stratified sampling by consensus prediction to reduce label skew.
    consensus_to_indices = {label: [] for label in LABELS}
    for i in range(total):
        row_a = df_a.iloc[i]
        row_b = df_b.iloc[i]
        avg_scores = {label: (float(row_a[label]) + float(row_b[label])) / 2.0 for label in LABELS}
        consensus = max(LABELS, key=lambda label: avg_scores[label])
        consensus_to_indices[consensus].append(i)

    per_label_target = max(1, n // len(LABELS))
    picked = []
    for label in LABELS:
        bucket = consensus_to_indices[label]
        if not bucket:
            continue
        k = min(per_label_target, len(bucket))
        picked.extend(random.sample(bucket, k))

    if len(picked) < n:
        remain = list(set(range(total)) - set(picked))
        extra_k = min(n - len(picked), len(remain))
        picked.extend(random.sample(remain, extra_k))

    if len(picked) > n:
        picked = random.sample(picked, n)

    return sorted(picked)


# ── annotation loop ───────────────────────────────────────────────────────────
def annotate(frames, timestamps, df_a, df_b, n_samples, sample_mode):
    total   = len(frames)
    indices = sample_indices(df_a, df_b, n_samples, total, mode=sample_mode)
    n       = len(indices)
    print(f"[annotation] {n} / {total} frames sampled ({sample_mode})\n")

    results = []
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1020, 720)

    for rank, idx in enumerate(indices, 1):
        pil_img = frames[idx]
        ts      = timestamps[idx]
        row_a   = df_a.iloc[idx]
        row_b   = df_b.iloc[idx]

        pred_a = top_pred(row_a, LABELS)
        pred_b = top_pred(row_b, LABELS)
        conf_a = float(row_a[pred_a])
        conf_b = float(row_b[pred_b])

        # build display frame
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if w > 980:
            bgr = cv2.resize(bgr, (980, int(h * 980 / w)))

        pw    = bgr.shape[1]
        panel = np.full((195, pw, 3), 18, dtype=np.uint8)

        def put(text, x, y, color=(210, 210, 210), scale=0.50, thick=1):
            cv2.putText(panel, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        # header
        put(f"Frame {rank}/{n}   t={ts:.2f}s", 8, 22)

        # Sys A — single prompt
        put("Sys A  (single prompt):", 8, 52, color=COL_A_HEAD, scale=0.48)
        x = 215
        for lb in LABELS:
            c   = float(row_a[lb])
            col = COL_HIT if lb == pred_a else COL_MISS
            put(f"{lb[:4]}:{c:.2f}", x, 52, color=col, scale=0.43)
            x += 133
        put(f"  =>  {pred_a.upper()}  ({conf_a:.3f})", 8, 74, color=COL_HIT)

        # Sys B — multi prompt
        put("Sys B  (multi prompt + max):", 8, 106, color=COL_B_HEAD, scale=0.48)
        x = 215
        for lb in LABELS:
            c   = float(row_b[lb])
            col = COL_HIT if lb == pred_b else COL_MISS
            put(f"{lb[:4]}:{c:.2f}", x, 106, color=col, scale=0.43)
            x += 133
        put(f"  =>  {pred_b.upper()}  ({conf_b:.3f})", 8, 128, color=COL_HIT)

        # key hint
        put("1=Running  2=Eating  3=Walking  4=Standing  5=Sitting  6=Lying   s=Skip  q=Quit",
            8, 170, color=COL_KEY, scale=0.41)

        display = np.vstack([bgr, panel])
        cv2.imshow(WINDOW_TITLE, display)

        while True:
            key    = cv2.waitKey(0) & 0xFF
            action = KEY_MAP.get(key)
            if action is None:
                continue

            if action == "quit":
                print("\n[annotation] quit — saving partial results ...")
                cv2.destroyAllWindows()
                return results

            if action == "skip":
                print(f"  [{rank}/{n}]  t={ts:.2f}s  SKIPPED")
                break

            human   = action
            agree_a = (pred_a == human)
            agree_b = (pred_b == human)
            print(f"  [{rank}/{n}]  t={ts:.2f}s  "
                  f"human={human:<10}"
                  f"A={pred_a}({'OK' if agree_a else '--'})  "
                  f"B={pred_b}({'OK' if agree_b else '--'})")
            results.append({
                "frame_idx":   idx,
                "timestamp_s": round(ts, 3),
                "human_label": human,
                "sysA_pred":   pred_a,
                "sysA_conf":   round(conf_a, 4),
                "sysA_agree":  int(agree_a),
                "sysB_pred":   pred_b,
                "sysB_conf":   round(conf_b, 4),
                "sysB_agree":  int(agree_b),
            })
            break

    cv2.destroyAllWindows()
    return results


# ── report ────────────────────────────────────────────────────────────────────
def save_and_report(results, out_dir, multi_reducer, sample_mode):
    if not results:
        print("[report] no annotations collected.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "manual_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")

    n       = len(results)
    a_right = sum(r["sysA_agree"] for r in results)
    b_right = sum(r["sysB_agree"] for r in results)

    from collections import defaultdict
    a_cls = defaultdict(lambda: [0, 0])
    b_cls = defaultdict(lambda: [0, 0])
    for r in results:
        h = r["human_label"]
        a_cls[h][1] += 1
        b_cls[h][1] += 1
        if r["sysA_agree"]: a_cls[h][0] += 1
        if r["sysB_agree"]: b_cls[h][0] += 1

    # classify labels into dynamic vs static
    dynamic = ["running", "walking"]
    static  = ["sitting", "lying", "standing"]
    eating  = ["eating"]

    def group_acc(cls_dict, group):
        ok = sum(cls_dict[l][0] for l in group if cls_dict[l][1] > 0)
        tot = sum(cls_dict[l][1] for l in group if cls_dict[l][1] > 0)
        return ok, tot

    lines = [
        "=" * 64,
        "Manual Annotation Spot-Check Report",
        "System A: Single Prompt (D1 only, 6 prompts)",
        f"System B: Multi Prompt  (5 variants + {multi_reducer}, 30 prompts)",
        f"Sampling: {sample_mode}",
        "=" * 64,
        f"Total annotated frames    : {n}",
        "",
        f"System A  (single prompt) : {a_right}/{n} = {a_right/n*100:.1f}%",
        f"System B  (multi  prompt) : {b_right}/{n} = {b_right/n*100:.1f}%",
        f"Improvement               : {(b_right-a_right)/n*100:+.1f} pp",
        "",
        f"  {'Label':<12} {'n':>4}  {'A ok':>5} {'A%':>5}  {'B ok':>5} {'B%':>5}",
        f"  {'-'*52}",
    ]
    for lb in LABELS:
        ca, ta = a_cls[lb]
        cb, tb = b_cls[lb]
        if ta == 0:
            lines.append(f"  {lb:<12} {'—':>4}  {'—':>5} {'—':>5}  {'—':>5} {'—':>5}")
        else:
            lines.append(
                f"  {lb:<12} {ta:>4}  "
                f"{ca:>5} {ca/ta*100:>4.0f}%  "
                f"{cb:>5} {cb/tb*100:>4.0f}%"
            )

    lines += ["", "  Group summary:"]
    for grp_name, grp in [("Dynamic (run/walk)", dynamic),
                           ("Static  (sit/lie/stand)", static),
                           ("Eating", eating)]:
        oa, ta = group_acc(a_cls, grp)
        ob, tb = group_acc(b_cls, grp)
        if ta == 0:
            lines.append(f"    {grp_name:<26}: A=—  B=—")
        else:
            lines.append(
                f"    {grp_name:<26}: "
                f"A={oa}/{ta}={oa/ta*100:.0f}%  "
                f"B={ob}/{tb}={ob/tb*100:.0f}%"
            )

    lines += [
        "",
        "Note: spot-check only — random sample, not a full evaluation.",
        "      Both systems use identical model weights and label space.",
        "      Difference reflects prompt multiplicity only.",
        "=" * 64,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    rpt = out_dir / "agreement_report.txt"
    rpt.write_text(report, encoding="utf-8")
    print(f"[saved] {rpt}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Spot-check: Single Prompt vs Multi Prompt human accuracy"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--n",    type=int,   default=50,
                        help="Frames to annotate (default: 50)")
    parser.add_argument("--fps",  type=float, default=2.0,
                        help="Sampling rate (default: 2.0)")
    parser.add_argument("--out",  default="ablation_output")
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument(
        "--multi-reducer",
        choices=["max", "mean"],
        default="max",
        help="Aggregation for System B prompts (default: max)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["random", "stratified"],
        default="random",
        help="Frame sampling mode for annotation (default: random)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)

    print("=== Loading video ===")
    reader     = VideoReader(args.video, sample_fps=args.fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames     = [img for _, img in frame_data]
    print(f"  {len(frames)} frames @ {args.fps} fps")

    print("\n=== Loading SigLIP model ===")
    clf = SigLIPClassifier(model_name=MODEL)
    clf._load()

    df_a, df_b = run_systems(clf, frames, timestamps, reducer=args.multi_reducer)

    print(f"\n=== Annotation (up to {args.n} frames) ===")
    print("Window opens for each frame.")
    print("Press 1-6 for your ground-truth label, s=skip, q=quit.\n")

    results = annotate(frames, timestamps, df_a, df_b, args.n, args.sample_mode)
    save_and_report(results, out_dir, args.multi_reducer, args.sample_mode)


if __name__ == "__main__":
    main()