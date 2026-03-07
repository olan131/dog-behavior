"""Gradio web UI for pet-behavior-clip.

Launch
------
::

    python ui/app.py

or via the CLI entry-point::

    pet-behavior-ui
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Allow importing the package from the project root when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import pandas as pd

from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.smoothing import smooth_scores
from pet_behavior_clip.anomaly import AnomalyDetector
from pet_behavior_clip.plots import (
    plot_behavior_timeline,
    plot_anomaly_heatmap,
    plot_confidence_distribution,
)
from pet_behavior_clip.report_llm import generate_report

logger = logging.getLogger(__name__)

_DEFAULT_LABELS = (
    "dog sitting calmly, dog walking normally, dog running, "
    "dog barking, dog scratching, dog shaking, dog limping, dog lying down"
)

_OUT_DIR = Path("ui_output")
_OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    video_file: Optional[str],
    labels_text: str,
    fps: float,
    smooth_window: int,
    smooth_method: str,
    anomaly_method: str,
    threshold: float,
    report_mode: str,
    model_name: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], str]:
    """Run the full pet-behaviour pipeline and return UI outputs.

    Returns
    -------
    summary_json, report_md, timeline_png, heatmap_png, dist_png, csv_path
    """
    if video_file is None:
        return (
            "錯誤：請先上傳影片檔案",
            "無報告 – 請上傳影片後重新分析",
            None,
            None,
            None,
            "",
        )

    label_list: List[str] = [l.strip() for l in labels_text.split(",") if l.strip()]
    stem = Path(video_file).stem

    # 1 – Frame sampling
    progress(0.0, desc="取樣影像幀 …")
    reader = VideoReader(video_file, sample_fps=fps)
    frame_data = reader.sample_frames()
    reader.release()
    timestamps = [t for t, _ in frame_data]
    frames = [img for _, img in frame_data]

    # 2 – Classification
    progress(0.2, desc="SigLIP 零樣本分類中 …")
    classifier = SigLIPClassifier(model_name=model_name)
    scores_df = classifier.classify_frames(frames, label_list, timestamps)

    # 3 – Smoothing
    progress(0.5, desc="時序平滑處理 …")
    smoothed_df = smooth_scores(scores_df, window=smooth_window, method=smooth_method)

    # 4 – Anomaly detection
    progress(0.65, desc="異常偵測中 …")
    detector = AnomalyDetector(method=anomaly_method, threshold=threshold)
    detected_df = detector.detect(smoothed_df)
    summary_dict = detector.summary(detected_df)

    # 5 – Save CSV
    csv_path = _OUT_DIR / f"{stem}_scores.csv"
    detected_df.to_csv(csv_path, index=False)

    # JSON summary
    json_path = _OUT_DIR / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6 – Plots
    progress(0.75, desc="繪製圖表 …")
    tl_path = _OUT_DIR / f"{stem}_timeline.png"
    hm_path = _OUT_DIR / f"{stem}_heatmap.png"
    dist_path = _OUT_DIR / f"{stem}_distribution.png"
    plot_behavior_timeline(detected_df, output_path=tl_path)
    plot_anomaly_heatmap(detected_df, output_path=hm_path)
    plot_confidence_distribution(detected_df, output_path=dist_path)

    # 7 – Report
    progress(0.90, desc="生成報告 …")
    report_path = _OUT_DIR / f"{stem}_report.md"
    report_md = generate_report(
        detected_df,
        label_list,
        video_path=video_file,
        mode=report_mode,
        output_path=report_path,
    )

    progress(1.0, desc="完成！")
    summary_text = json.dumps(summary_dict, indent=2, ensure_ascii=False)
    return (
        summary_text,
        report_md,
        str(tl_path),
        str(hm_path),
        str(dist_path),
        str(csv_path),
    )


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pet Behavior Clip – 寵物行為分析系統") as demo:
        gr.Markdown(
            """# 🐾 Pet Behavior Clip
            **基於 SigLIP 零樣本分類的寵物行為異常偵測系統**
            
            上傳寵物影片，輸入行為標籤，系統將自動分析行為模式並標記異常片段。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 輸入設定")
                video_input = gr.Video(label="上傳影片", sources=["upload"])
                labels_input = gr.Textbox(
                    label="行為標籤（逗號分隔）",
                    value=_DEFAULT_LABELS,
                    lines=3,
                )
                fps_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="取樣頻率 (fps)",
                )
                model_input = gr.Textbox(
                    label="HuggingFace 模型",
                    value="google/siglip-so400m-patch14-224",
                )

                gr.Markdown("### ⚙️ 分析參數")
                with gr.Row():
                    smooth_window = gr.Slider(3, 15, step=2, value=5, label="平滑視窗大小")
                    smooth_method = gr.Dropdown(
                        choices=["rolling_mean", "gaussian", "exponential"],
                        value="rolling_mean",
                        label="平滑方法",
                    )
                with gr.Row():
                    anomaly_method = gr.Dropdown(
                        choices=["zscore", "iqr"],
                        value="zscore",
                        label="異常偵測方法",
                    )
                    threshold = gr.Slider(1.0, 5.0, step=0.5, value=2.5, label="異常閾值")
                report_mode = gr.Radio(
                    choices=["template", "llm"],
                    value="template",
                    label="報告模式（llm 需要 OPENAI_API_KEY）",
                )
                run_btn = gr.Button("🚀 開始分析", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 分析結果")
                with gr.Tabs():
                    with gr.Tab("摘要 & 報告"):
                        summary_out = gr.Code(label="偵測摘要 (JSON)", language="json")
                        report_out = gr.Markdown(label="分析報告")
                    with gr.Tab("行為時間線"):
                        timeline_out = gr.Image(label="行為信心度時間線")
                    with gr.Tab("熱力圖"):
                        heatmap_out = gr.Image(label="行為分數熱力圖")
                    with gr.Tab("分佈圖"):
                        dist_out = gr.Image(label="信心度分佈")
                    with gr.Tab("CSV 下載"):
                        csv_out = gr.File(label="下載原始分數 CSV")

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
                report_mode,
                model_input,
            ],
            outputs=[
                summary_out,
                report_out,
                timeline_out,
                heatmap_out,
                dist_out,
                csv_out,
            ],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)


def main() -> None:
    """Entry-point for the ``pet-behavior-ui`` console script."""
    logging.basicConfig(level=logging.INFO)
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
