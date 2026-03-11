"""Gradio web UI for pet-behavior-clip.

Launch with:
    python ui/app.py
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from pet_behavior_clip.anomaly import AnomalyDetector
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.contextual import aggregate_sequence_scores, compute_ece_from_labeled_scores
from pet_behavior_clip.prompt import classify_with_template_max, classify_with_single_prompt
from pet_behavior_clip.behavior_postprocess import (
    build_behavior_segments,
    infer_frame_behaviors,
    smooth_behavior_labels,
    summarize_behavior_results,
)
from pet_behavior_clip.plots import (
    plot_anomaly_heatmap,
    plot_behavior_timeline,
    plot_behavior_segments_timeline,
    plot_confidence_distribution,
)
from pet_behavior_clip.report import generate_report
from pet_behavior_clip.smoothing import smooth_scores
from pet_behavior_clip.video import VideoReader

logger = logging.getLogger(__name__)

_DEFAULT_LABELS = (
    "running, eating, walking, standing, sitting, lying"
)

_OUT_DIR = Path("ui_output")
_OUT_DIR.mkdir(exist_ok=True)

_DEFAULT_SERVER_HOST = "0.0.0.0"
_DEFAULT_SERVER_PORT = 7860
_PORT_SEARCH_SPAN = 40


def run_analysis(
    video_file: Optional[str],
    labels_text: str,
    fps: float,
    smooth_window: int,
    smooth_method: str,
    anomaly_method: str,
    threshold: float,
    model_name: str,
    sequence_aggregate: str,
    sequence_window: int,
    confidence_threshold: float,
    margin_threshold: float,
    label_smooth_seconds: float,
    anomaly_alert_threshold: float,
    prompt_mode: str = "Multi-prompt (5 variants + max)",
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[
    Optional[str],
    str,
    str,
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    str,
    str,
]:
    """Run local analysis and return UI outputs."""
    if video_file is None:
        return None, "{}", "", "", None, None, None, "", "", ""

    labels: List[str] = [item.strip() for item in labels_text.split(",") if item.strip()]
    stem = Path(video_file).stem

    progress(0.0, desc="Sampling frames...")
    reader = VideoReader(video_file, sample_fps=fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames = [img for _, img in frame_data]

    progress(0.2, desc="Running SigLIP inference...")
    classifier = SigLIPClassifier(model_name=model_name)
    if prompt_mode == "Single prompt (D1 only)":
        scores_df = classify_with_single_prompt(
            classifier=classifier,
            frames=frames,
            labels=labels,
            timestamps=timestamps,
        )
    else:
        scores_df = classify_with_template_max(
            classifier=classifier,
            frames=frames,
            labels=labels,
            timestamps=timestamps,
        )

    scores_df = aggregate_sequence_scores(
        scores_df,
        mode=sequence_aggregate,
        window=sequence_window,
    )

    progress(0.5, desc="Smoothing scores...")
    smoothed_df = smooth_scores(scores_df, window=smooth_window, method=smooth_method)

    progress(0.65, desc="Detecting anomalies...")
    detector = AnomalyDetector(method=anomaly_method, threshold=threshold)
    detected_df = detector.detect(smoothed_df)
    summary_dict = detector.summary(detected_df)
    summary_dict["sequence_aggregate"] = sequence_aggregate
    summary_dict["sequence_window"] = sequence_window
    summary_dict["ece"] = compute_ece_from_labeled_scores(detected_df)

    csv_path = _OUT_DIR / f"{stem}_scores.csv"
    detected_df.to_csv(csv_path, index=False)

    json_path = _OUT_DIR / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    progress(0.8, desc="Rendering plots...")
    score_tl_path = _OUT_DIR / f"{stem}_timeline.png"
    hm_path = _OUT_DIR / f"{stem}_heatmap.png"
    dist_path = _OUT_DIR / f"{stem}_distribution.png"
    plot_behavior_timeline(detected_df, output_path=score_tl_path)
    plot_anomaly_heatmap(detected_df, output_path=hm_path)
    plot_confidence_distribution(detected_df, output_path=dist_path)

    progress(0.95, desc="Building local report...")
    report_path = _OUT_DIR / f"{stem}_report.md"
    report_md = generate_report(
        detected_df,
        labels,
        video_path=video_file,
        output_path=report_path,
    )

    progress(0.98, desc="Post-processing behavior labels...")
    labeled_df = infer_frame_behaviors(
        detected_df,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    labeled_df = smooth_behavior_labels(
        labeled_df,
        window_seconds=label_smooth_seconds,
        label_col="behavior_label",
        output_col="behavior_smooth",
    )
    segments_df = build_behavior_segments(
        labeled_df,
        label_col="behavior_smooth",
    )

    behavior_tl_path = _OUT_DIR / f"{stem}_behavior_timeline.png"
    plot_behavior_segments_timeline(segments_df, output_path=behavior_tl_path)

    behavior_md = summarize_behavior_results(
        labeled_df,
        segments_df,
        anomaly_alert_threshold=anomaly_alert_threshold,
        label_col="behavior_smooth",
    )

    labeled_csv_path = _OUT_DIR / f"{stem}_labels.csv"
    labeled_df.to_csv(labeled_csv_path, index=False)

    segments_csv_path = _OUT_DIR / f"{stem}_segments.csv"
    segments_df.to_csv(segments_csv_path, index=False)

    progress(1.0, desc="Done")
    summary_text = json.dumps(summary_dict, indent=2, ensure_ascii=False)
    return (
        str(behavior_tl_path),
        summary_text,
        report_md,
        behavior_md,
        str(score_tl_path),
        str(hm_path),
        str(dist_path),
        str(csv_path),
        str(labeled_csv_path),
        str(segments_csv_path),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pet Behavior Clip") as demo:
        gr.Markdown(
            """# Pet Behavior Clip
Local and privacy-focused behavior analysis:
Video -> SigLIP -> Temporal smoothing -> Anomaly detection.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input video", sources=["upload"])
                labels_input = gr.Textbox(
                    label="Behavior labels (comma separated)",
                    value=_DEFAULT_LABELS,
                    lines=2,
                )
                fps_slider = gr.Slider(minimum=0.5, maximum=5.0, step=0.5, value=1.0, label="Sampling FPS")
                model_input = gr.Textbox(
                    label="HuggingFace model",
                    value="google/siglip-so400m-patch14-224",
                )

                smooth_window = gr.Slider(3, 15, step=2, value=5, label="Smoothing window")
                smooth_method = gr.Dropdown(
                    choices=["rolling_mean", "gaussian", "exponential"],
                    value="rolling_mean",
                    label="Smoothing method",
                )
                anomaly_method = gr.Dropdown(
                    choices=["zscore", "iqr"],
                    value="zscore",
                    label="Anomaly method",
                )
                threshold = gr.Slider(1.0, 5.0, step=0.5, value=2.5, label="Anomaly threshold")
                sequence_aggregate = gr.Dropdown(
                    choices=["none", "prob", "logit"],
                    value="none",
                    label="Sequence aggregation",
                )
                sequence_window = gr.Slider(3, 21, step=2, value=5, label="Sequence window")
                confidence_threshold = gr.Slider(
                    minimum=0.2,
                    maximum=0.9,
                    step=0.05,
                    value=0.35,
                    label="Behavior confidence threshold",
                )
                margin_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=0.3,
                    step=0.05,
                    value=0.10,
                    label="Margin threshold",
                )
                label_smooth_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.5,
                    value=2.0,
                    label="Behavior smoothing window (+/- seconds)",
                )
                anomaly_alert_threshold = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=2.0,
                    label="Anomaly alert threshold",
                )
                prompt_mode = gr.Radio(
                    choices=[
                        "Multi-prompt (5 variants + max)",
                        "Single prompt (D1 only)",
                    ],
                    value="Multi-prompt (5 variants + max)",
                    label="Prompt mode",
                    info="Multi-prompt: 5 descriptions per class, better for dynamic actions. Single prompt: 1 description per class, faster.",
                )
                run_btn = gr.Button("Run analysis", variant="primary")

            with gr.Column(scale=2):
                behavior_timeline_out = gr.Image(label="Behavior Timeline (Segments)")
                summary_out = gr.Code(label="Summary JSON", language="json")
                report_out = gr.Markdown(label="Report")
                behavior_out = gr.Markdown(label="Behavior Timeline Summary")
                timeline_out = gr.Image(label="Score Timeline")
                heatmap_out = gr.Image(label="Heatmap")
                dist_out = gr.Image(label="Distribution")
                csv_out = gr.File(label="Scores CSV")
                labels_csv_out = gr.File(label="Behavior Labels CSV")
                segments_csv_out = gr.File(label="Behavior Segments CSV")

        run_btn.click(
            fn=run_analysis,
            inputs=[
                video_input,
                labels_input,
                fps_slider,
                smooth_window,
                smooth_method,
                anomaly_method,
                threshold,
                model_input,
                sequence_aggregate,
                sequence_window,
                confidence_threshold,
                margin_threshold,
                label_smooth_seconds,
                anomaly_alert_threshold,
                prompt_mode,
            ],
            outputs=[
                behavior_timeline_out,
                summary_out,
                report_out,
                behavior_out,
                timeline_out,
                heatmap_out,
                dist_out,
                csv_out,
                labels_csv_out,
                segments_csv_out,
            ],
        )

    return demo


def _is_port_available(host: str, port: int) -> bool:
    """Return True when a TCP port is available on this machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _resolve_launch_port(preferred_port: int) -> int:
    """Pick preferred port when free, otherwise scan nearby ports."""
    if _is_port_available(_DEFAULT_SERVER_HOST, preferred_port):
        return preferred_port

    for port in range(preferred_port + 1, preferred_port + 1 + _PORT_SEARCH_SPAN):
        if _is_port_available(_DEFAULT_SERVER_HOST, port):
            logger.warning(
                "Port %s is in use; using available port %s instead.",
                preferred_port,
                port,
            )
            return port

    raise OSError(
        f"No available port found in range {preferred_port}-{preferred_port + _PORT_SEARCH_SPAN}."
    )


def _launch_ui() -> None:
    """Launch UI with automatic port fallback when preferred port is occupied."""
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", str(_DEFAULT_SERVER_PORT)))
    launch_port = _resolve_launch_port(preferred_port)

    app = build_ui()
    app.launch(server_name=_DEFAULT_SERVER_HOST, server_port=launch_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _launch_ui()


def main() -> None:
    """Entry-point for the ``pet-behavior-ui`` console script."""
    logging.basicConfig(level=logging.INFO)
    _launch_ui()
