"""manual_label.py - Compare two inference modes against human annotation.

System A: cli.py style  - raw labels passed directly to classify_frames
System B: template mode - 5 prompt variants per label + max aggregation

Each frame is shown once. You enter your ground-truth label once.
Both systems are evaluated against your answer.

Usage:
    python manual_label.py --video dog.mp4
    python manual_label.py --video dog.mp4 --n 30 --fps 2.0

Keys:
    1  -> Active
    2  -> Resting
    3  -> Eating/Drinking
    s  -> Skip
    q  -> Quit and save
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# -- Auto-detect project root --
_here = Path(__file__).resolve().parent
if _here.name == "pet_behavior_clip":
    _root = _here.parent
else:
    _root = _here
sys.path.insert(0, str(_root))

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.prompt import (
    build_label_prompt_result,
    flatten_prompt_map,
    aggregate_prompt_scores,
)

# -- System A: cli.py default labels (raw strings) --
LABELS_A = [
    "a picture of an animal moving",
    "a picture of an animal eating",
    "a picture of an animal resting",
]

# -- System B: template mode labels (will be expanded to 5 variants each) --
LABELS_B = [
    "active",
    "resting",
    "eating/drinking",
]

# Mapping from System B labels back to display names matching System A
DISPLAY = {
    "a picture of an animal moving":  "Moving",
    "a picture of an animal eating":  "Eating",
    "a picture of an animal resting": "Resting",
    "active":          "Active",
    "resting":         "Resting",
    "eating/drinking": "Eating/Drinking",
}

MODEL = "google/siglip-so400m-patch14-224"
CTX   = "overhead top-down surveillance camera"
KEY_MAP = {
    ord("1"): "moving/active",
    ord("2"): "resting",
    ord("3"): "eating/drinking",
    ord("s"): "skip",
    ord("q"): "quit",
}
WINDOW = "1=Moving/Active  2=Resting  3=Eating/Drinking  s=Skip  q=Quit"


# ====================================================================
# Inference
# ====================================================================
def run_system_a(clf, frames, timestamps):
    """System A: raw labels, direct classify_frames (cli.py style)."""
    print("[System A] running inference (raw labels) ...")
    df = clf.classify_frames(frames, LABELS_A, timestamps)
    return df


def run_system_b(clf, frames, timestamps):
    """System B: template expansion + max aggregation."""
    print("[System B] running inference (template + max) ...")
    res = build_label_prompt_result(labels=LABELS_B, mode="template", camera_context=CTX)
    pm  = res["prompt_map"]
    pl  = flatten_prompt_map(pm)
    raw = clf.classify_frames(frames, pl, timestamps)
    df  = aggregate_prompt_scores(raw, pm, reducer="max")
    return df


def top_label_a(row):
    return max(LABELS_A, key=lambda l: row[l])


def top_label_b(row):
    return max(LABELS_B, key=lambda l: row[l])


def normalize_human(human: str, sys_label: str) -> bool:
    """Check if human label matches system prediction."""
    h = human.lower()
    # System A labels
    if sys_label in LABELS_A:
        if sys_label == "a picture of an animal moving"  and h == "moving/active":  return True
        if sys_label == "a picture of an animal eating"  and h == "eating/drinking": return True
        if sys_label == "a picture of an animal resting" and h == "resting":         return True
        return False
    # System B labels
    if sys_label == "active"          and h == "moving/active":   return True
    if sys_label == "resting"         and h == "resting":         return True
    if sys_label == "eating/drinking" and h == "eating/drinking": return True
    return False


# ====================================================================
# Annotation loop
# ====================================================================
def annotate(frames, timestamps, df_a, df_b, n_samples, out_dir):
    total   = len(frames)
    n       = min(n_samples, total)
    indices = sorted(random.sample(range(total), n))
    print(f"[annotation] {n} frames randomly sampled from {total}")

    results = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 640)

    for rank, idx in enumerate(indices, 1):
        pil_img = frames[idx]
        ts      = timestamps[idx]
        row_a   = df_a.iloc[idx]
        row_b   = df_b.iloc[idx]

        top_a   = top_label_a(row_a)
        top_b   = top_label_b(row_b)
        conf_a  = float(row_a[top_a])
        conf_b  = float(row_b[top_b])
        disp_a  = DISPLAY.get(top_a, top_a)
        disp_b  = DISPLAY.get(top_b, top_b)

        # Build display image
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if w > 920:
            bgr = cv2.resize(bgr, (920, int(h * 920 / w)))

        # Info panel
        ph    = 140
        panel = np.full((ph, bgr.shape[1], 3), 25, dtype=np.uint8)

        def put(text, x, y, color=(220, 220, 220), scale=0.52, thick=1):
            cv2.putText(panel, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        put(f"Frame {rank}/{n}   t={ts:.1f}s", 10, 22)

        # System A scores
        put("Sys A (raw):", 10, 50, color=(100, 180, 255))
        x = 130
        for l in LABELS_A:
            c   = float(row_a[l])
            col = (0, 220, 80) if l == top_a else (140, 140, 140)
            put(f"{DISPLAY[l]}:{c:.2f}", x, 50, color=col)
            x += 190
        put(f"=> {disp_a} ({conf_a:.3f})", 10, 74, color=(0, 220, 80))

        # System B scores
        put("Sys B (tmpl):", 10, 98, color=(255, 180, 80))
        x = 130
        for l in LABELS_B:
            c   = float(row_b[l])
            col = (0, 220, 80) if l == top_b else (140, 140, 140)
            put(f"{DISPLAY[l]}:{c:.2f}", x, 98, color=col)
            x += 190
        put(f"=> {disp_b} ({conf_b:.3f})", 10, 122, color=(255, 180, 80))

        put("Your label:  1=Moving/Active  2=Resting  3=Eating/Drinking  s=Skip  q=Quit",
            10, 136, color=(200, 200, 60), scale=0.45)

        display = np.vstack([bgr, panel])
        cv2.imshow(WINDOW, display)

        # Wait for key
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key not in KEY_MAP:
                continue
            action = KEY_MAP[key]

            if action == "quit":
                print("\n[annotation] quit by user, saving partial results ...")
                cv2.destroyAllWindows()
                return results

            if action == "skip":
                print(f"  [{rank}/{n}] t={ts:.1f}s  SKIPPED")
                break

            human = action  # "moving/active" | "resting" | "eating/drinking"
            agree_a = normalize_human(human, top_a)
            agree_b = normalize_human(human, top_b)
            tag_a = "OK" if agree_a else "WRONG"
            tag_b = "OK" if agree_b else "WRONG"
            print(f"  [{rank}/{n}] t={ts:.1f}s  "
                  f"human={human}  "
                  f"A={disp_a}({tag_a})  "
                  f"B={disp_b}({tag_b})")
            results.append({
                "frame_idx":   idx,
                "timestamp_s": round(ts, 2),
                "human_label": human,
                "sysA_label":  top_a,
                "sysA_conf":   round(conf_a, 4),
                "sysA_agree":  int(agree_a),
                "sysB_label":  top_b,
                "sysB_conf":   round(conf_b, 4),
                "sysB_agree":  int(agree_b),
            })
            break

    cv2.destroyAllWindows()
    return results


# ====================================================================
# Report
# ====================================================================
def save_and_report(results, out_dir):
    if not results:
        print("[report] no annotations collected.")
        return

    csv_path = out_dir / "manual_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")

    n       = len(results)
    a_right = sum(r["sysA_agree"] for r in results)
    b_right = sum(r["sysB_agree"] for r in results)

    # Per-class breakdown
    from collections import defaultdict
    a_cls = defaultdict(lambda: [0, 0])  # [correct, total]
    b_cls = defaultdict(lambda: [0, 0])
    for r in results:
        h = r["human_label"]
        a_cls[h][1] += 1
        b_cls[h][1] += 1
        if r["sysA_agree"]: a_cls[h][0] += 1
        if r["sysB_agree"]: b_cls[h][0] += 1

    lines = [
        "=" * 56,
        "Manual Annotation Comparison Report",
        "=" * 56,
        f"Total annotated frames   : {n}",
        "",
        f"System A (raw labels)    : {a_right}/{n} = {a_right/n*100:.1f}%",
        f"System B (template+max)  : {b_right}/{n} = {b_right/n*100:.1f}%",
        "",
        "Per human-label breakdown:",
    ]

    for h in ["moving/active", "resting", "eating/drinking"]:
        ca, ta = a_cls[h]
        cb, tb = b_cls[h]
        if ta == 0:
            lines.append(f"  {h:20s}: no samples")
        else:
            lines.append(
                f"  {h:20s}: A={ca}/{ta}={ca/ta*100:.0f}%  "
                f"B={cb}/{tb}={cb/tb*100:.0f}%"
            )

    lines += [
        "",
        "Note: spot-check only (random sample), not a full evaluation.",
        "=" * 56,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    rpt = out_dir / "agreement_report.txt"
    rpt.write_text(report, encoding="utf-8")
    print(f"[saved] {rpt}")


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Compare two inference modes via manual annotation")
    parser.add_argument("--video", required=True)
    parser.add_argument("--n",    type=int,   default=30)
    parser.add_argument("--fps",  type=float, default=2.0)
    parser.add_argument("--out",  default="ablation_output")
    parser.add_argument("--seed", type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading video ===")
    reader     = VideoReader(args.video, sample_fps=args.fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames     = [img for _, img in frame_data]
    print(f"  {len(frames)} frames @ {args.fps} fps")

    print("\n=== Loading model ===")
    clf = SigLIPClassifier(model_name=MODEL)
    clf._load()

    df_a = run_system_a(clf, frames, timestamps)
    df_b = run_system_b(clf, frames, timestamps)

    print(f"\n=== Annotation ({args.n} frames) ===")
    print("Window will open. Each frame shows both system predictions.")
    print("Press 1/2/3 to enter your label.\n")

    results = annotate(frames, timestamps, df_a, df_b, args.n, out_dir)
    save_and_report(results, out_dir)


if __name__ == "__main__":
    main()
