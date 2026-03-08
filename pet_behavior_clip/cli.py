"""cli.py – Command-line interface for pet-behavior-clip.

Usage examples
--------------
Analyse a video with default labels::

    python -m pet_behavior_clip.cli analyze my_dog.mp4

Custom labels and output directory::

    python -m pet_behavior_clip.cli analyze my_dog.mp4 \\
        --labels "dog sitting,dog running,dog barking,dog eating" \\
        --output-dir ./results \\
        --fps 2 \\
        --smooth-window 7 \\
        --anomaly-method zscore \\
        --threshold 2.5

Generate an LLM report (requires OPENROUTER_API_KEY)::

    python -m pet_behavior_clip.cli analyze my_dog.mp4 --report-mode llm
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

_DEFAULT_LABELS = [
    "dog sitting calmly",
    "dog walking normally",
    "dog running",
    "dog barking",
    "dog scratching",
    "dog shaking",
    "dog limping",
    "dog lying down",
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
    "--report-mode",
    default="template",
    show_default=True,
    type=click.Choice(["template", "llm"]),
    help="Report generation mode.",
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
    "--prompt-mode",
    default="template",
    show_default=True,
    type=click.Choice(["off", "template", "llm"]),
    help="Prompt generation mode for zero-shot classification.",
)
@click.option(
    "--prompt-llm-model",
    default="openai/gpt-4o-mini",
    show_default=True,
    help="OpenRouter model used when --prompt-mode=llm.",
)
@click.option(
    "--openrouter-api-key",
    default="",
    show_default=False,
    help="OpenRouter API key passed directly (overrides environment variable).",
)
@click.option(
    "--prompt-aggregate",
    default="max",
    show_default=True,
    type=click.Choice(["max", "mean"]),
    help="How to aggregate multiple prompt scores into one class score.",
)
@click.option(
    "--context-mode",
    default="off",
    show_default=True,
    type=click.Choice(["off", "daynight"]),
    help="Context-aware scoring mode. 'daynight' mixes daytime/nighttime prompt scores.",
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
    report_mode: str,
    output_dir: str,
    model: str,
    prompt_mode: str,
    prompt_llm_model: str,
    openrouter_api_key: str,
    prompt_aggregate: str,
    context_mode: str,
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
    click.echo("\n[1/5] Sampling frames …")
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
    click.echo("[2/5] Running SigLIP zero-shot classification …")
    from pet_behavior_clip.clip_zeroshot import SigLIPClassifier

    from pet_behavior_clip.prompt_llm import (
        aggregate_prompt_scores,
        build_label_prompt_result,
        flatten_prompt_map,
    )
    from pet_behavior_clip.contextual import (
        add_context_suffix,
        aggregate_sequence_scores,
        compute_ece_from_labeled_scores,
        estimate_night_probability,
        mix_day_night_scores,
    )

    prompt_result = build_label_prompt_result(
        labels=label_list,
        mode=prompt_mode,
        llm_model=prompt_llm_model,
        llm_api_key=openrouter_api_key or None,
    )
    prompt_map = prompt_result["prompt_map"]
    prompt_list = flatten_prompt_map(prompt_map)
    click.echo(
        f"       → Prompt mode: {prompt_mode} | {len(prompt_list)} prompts for {len(label_list)} classes"
    )
    if bool(prompt_result.get("fallback_used", False)):
        reason = str(prompt_result.get("fallback_reason") or "unknown")
        click.echo(
            "       → WARNING: LLM prompt generation failed, fallback to template "
            f"(reason: {reason})"
        )

    classifier = SigLIPClassifier(model_name=model)

    if context_mode == "daynight":
        day_map = add_context_suffix(prompt_map, "captured during daytime")
        night_map = add_context_suffix(prompt_map, "captured during nighttime")

        day_prompts = flatten_prompt_map(day_map)
        night_prompts = flatten_prompt_map(night_map)

        day_prompt_scores = classifier.classify_frames(frames, day_prompts, timestamps)
        night_prompt_scores = classifier.classify_frames(frames, night_prompts, timestamps)

        day_scores = aggregate_prompt_scores(day_prompt_scores, day_map, reducer=prompt_aggregate)
        night_scores = aggregate_prompt_scores(night_prompt_scores, night_map, reducer=prompt_aggregate)

        p_night = estimate_night_probability(frames)
        scores_df = mix_day_night_scores(day_scores, night_scores, p_night)
        click.echo("       → Context mode: daynight (brightness-weighted mixing)")
    else:
        prompt_scores_df = classifier.classify_frames(frames, prompt_list, timestamps)
        scores_df = aggregate_prompt_scores(
            prompt_scores_df,
            prompt_map,
            reducer=prompt_aggregate,
        )

    scores_df = aggregate_sequence_scores(
        scores_df,
        mode=sequence_aggregate,
        window=sequence_window,
    )
    click.echo(
        f"       → Sequence aggregation: {sequence_aggregate} (window={sequence_window})"
    )
    click.echo(f"       → Scores shape: {scores_df.shape}")

    # ------------------------------------------------------------------
    # 3. Temporal smoothing
    # ------------------------------------------------------------------
    click.echo("[3/5] Applying temporal smoothing …")
    from pet_behavior_clip.smoothing import smooth_scores

    smoothed_df = smooth_scores(scores_df, window=smooth_window, method=smooth_method)

    # ------------------------------------------------------------------
    # 4. Anomaly detection
    # ------------------------------------------------------------------
    click.echo("[4/5] Detecting anomalies …")
    from pet_behavior_clip.anomaly import AnomalyDetector

    detector = AnomalyDetector(method=anomaly_method, threshold=threshold)
    detected_df = detector.detect(smoothed_df)
    summary = detector.summary(detected_df)
    summary["context_mode"] = context_mode
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
    click.echo("[5/5] Saving outputs …")

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

    # Report
    from pet_behavior_clip.report_llm import generate_report

    report_md = generate_report(
        detected_df,
        label_list,
        video_path=video_path,
        mode=report_mode,
        output_path=out_dir / f"{stem}_report.md",
        llm_api_key=openrouter_api_key or None,
    )
    click.echo(f"       Report → {out_dir}/{stem}_report.md")

    click.echo("\n✅  Analysis complete.")
    click.echo("\n--- Report Preview ---")
    click.echo(report_md[:800] + ("…" if len(report_md) > 800 else ""))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
