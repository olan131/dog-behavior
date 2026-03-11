"""cli.py - Command-line interface for pet-behavior-clip.

The CLI runs a fully local pipeline:
video -> SigLIP scoring -> temporal smoothing -> anomaly detection -> outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

_DEFAULT_LABELS = [
    "running",
    "eating",
    "walking",
    "standing",
    "sitting",
    "lying",
]


@click.group()
@click.version_option(package_name="pet-behavior-clip")
def cli() -> None:
    """pet-behavior-clip – Pet behaviour anomaly detection via SigLIP."""


@cli.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--labels",
    default=",".join(_DEFAULT_LABELS),
    show_default=True,
    help="Comma-separated list of behaviour labels for zero-shot classification.",
)
@click.option(
    "--fps",
    default=1.0,
    show_default=True,
    type=float,
    help="Number of frames to sample per second.",
)
@click.option(
    "--smooth-window",
    default=5,
    show_default=True,
    type=int,
    help="Temporal smoothing window size (frames).",
)
@click.option(
    "--smooth-method",
    default="rolling_mean",
    show_default=True,
    type=click.Choice(["rolling_mean", "gaussian", "exponential"]),
    help="Smoothing algorithm.",
)
@click.option(
    "--anomaly-method",
    default="zscore",
    show_default=True,
    type=click.Choice(["zscore", "iqr"]),
    help="Anomaly detection algorithm.",
)
@click.option(
    "--threshold",
    default=2.5,
    show_default=True,
    type=float,
    help="Anomaly detection threshold.",
)
@click.option(
    "--output-dir",
    default="ui_output",
    show_default=True,
    type=click.Path(),
    help="Directory for output files (CSV, JSON, PNG, MD).",
)
@click.option(
    "--model",
    default="google/siglip-so400m-patch14-224",
    show_default=True,
    help="HuggingFace model identifier.",
)
@click.option(
    "--sequence-aggregate",
    default="none",
    show_default=True,
    type=click.Choice(["none", "prob", "logit"]),
    help="Temporal sequence aggregation mode before smoothing.",
)
@click.option(
    "--sequence-window",
    default=5,
    show_default=True,
    type=int,
    help="Window size used by sequence aggregation modes.",
)
def analyze(
    video_path: str,
    labels: str,
    fps: float,
    smooth_window: int,
    smooth_method: str,
    anomaly_method: str,
    threshold: float,
    output_dir: str,
    model: str,
    sequence_aggregate: str,
    sequence_window: int,
) -> None:
    """Analyse VIDEO_PATH for pet behaviour anomalies."""
    label_list: List[str] = [l.strip() for l in labels.split(",") if l.strip()]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    click.echo(f"📹  Video      : {video_path}")
    click.echo(f"🏷   Labels     : {label_list}")
    click.echo(f"📂  Output dir : {out_dir}")

    # ------------------------------------------------------------------
    # 1. Frame sampling
    # ------------------------------------------------------------------
    click.echo("\n[1/5] Sampling frames ...")
    from pet_behavior_clip.video import VideoReader

    reader = VideoReader(video_path, sample_fps=fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames = [img for _, img in frame_data]
    click.echo(f"       → {len(frames)} frames sampled.")

    # ------------------------------------------------------------------
    # 2. Zero-shot classification
    # ------------------------------------------------------------------
    click.echo("[2/5] Running SigLIP zero-shot classification ...")
    from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
    from pet_behavior_clip.contextual import (
        aggregate_sequence_scores,
        compute_ece_from_labeled_scores,
    )
    from pet_behavior_clip.prompt import classify_with_template_max

    classifier = SigLIPClassifier(model_name=model)
    scores_df = classify_with_template_max(
        classifier=classifier,
        frames=frames,
        labels=label_list,
        timestamps=timestamps,
    )

    scores_df = aggregate_sequence_scores(
        scores_df,
        mode=sequence_aggregate,
        window=sequence_window,
    )
    click.echo(
        f"       -> Sequence aggregation: {sequence_aggregate} (window={sequence_window})"
    )
    click.echo(f"       → Scores shape: {scores_df.shape}")

    # ------------------------------------------------------------------
    # 3. Temporal smoothing
    # ------------------------------------------------------------------
    click.echo("[3/5] Applying temporal smoothing ...")
    from pet_behavior_clip.smoothing import smooth_scores

    smoothed_df = smooth_scores(scores_df, window=smooth_window, method=smooth_method)

    # ------------------------------------------------------------------
    # 4. Anomaly detection
    # ------------------------------------------------------------------
    click.echo("[4/5] Detecting anomalies ...")
    from pet_behavior_clip.anomaly import AnomalyDetector

    detector = AnomalyDetector(method=anomaly_method, threshold=threshold)
    detected_df = detector.detect(smoothed_df)
    summary = detector.summary(detected_df)
    summary["sequence_aggregate"] = sequence_aggregate
    summary["sequence_window"] = sequence_window
    summary["ece"] = compute_ece_from_labeled_scores(detected_df)
    click.echo(
        f"       → {summary['anomaly_frames']}/{summary['total_frames']} frames anomalous "
        f"({summary['anomaly_ratio']:.1%})."
    )

    # ------------------------------------------------------------------
    # 5. Outputs
    # ------------------------------------------------------------------
    click.echo("[5/5] Saving outputs ...")

    # CSV
    csv_path = out_dir / f"{stem}_scores.csv"
    detected_df.to_csv(csv_path, index=False)
    click.echo(f"       CSV  → {csv_path}")

    # JSON summary
    json_path = out_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"       JSON → {json_path}")

    # Plots
    from pet_behavior_clip.plots import (
        plot_behavior_timeline,
        plot_anomaly_heatmap,
        plot_confidence_distribution,
    )

    plot_behavior_timeline(detected_df, output_path=out_dir / f"{stem}_timeline.png")
    plot_anomaly_heatmap(detected_df, output_path=out_dir / f"{stem}_heatmap.png")
    plot_confidence_distribution(detected_df, output_path=out_dir / f"{stem}_distribution.png")
    click.echo(f"       Plots → {out_dir}/")

    # Local report
    from pet_behavior_clip.report import generate_report

    report_md = generate_report(
        detected_df,
        label_list,
        video_path=video_path,
        output_path=out_dir / f"{stem}_report.md",
    )
    click.echo(f"       Report → {out_dir}/{stem}_report.md")

    click.echo("\n✅  Analysis complete.")
    click.echo("\n--- Report Preview ---")
    click.echo(report_md[:800] + ("…" if len(report_md) > 800 else ""))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
