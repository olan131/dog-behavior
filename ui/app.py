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
import socket
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
    "dog barking, dog scratching, dog shaking, dog lying down"
)

_OUT_DIR = Path("ui_output")
_OUT_DIR.mkdir(exist_ok=True)

_DEFAULT_SERVER_HOST = "0.0.0.0"
_DEFAULT_SERVER_PORT = 7860
_PORT_SEARCH_SPAN = 40


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


_INTERPRET_THRESHOLD = 0.10


def _build_readable_summary(
    detected_df: pd.DataFrame,
    labels: List[str],
    threshold: float = _INTERPRET_THRESHOLD,
    max_rows: int = 20,
) -> str:
    """Return a concise, human-readable interpretation of frame scores."""
    if detected_df.empty:
        return "## 可讀摘要\n\n無可分析資料。"

    available_labels = [l for l in labels if l in detected_df.columns]
    if not available_labels:
        return "## 可讀摘要\n\n找不到可用的行為分數欄位。"

    score_frame = detected_df[available_labels]
    top_label = score_frame.idxmax(axis=1)
    top_conf = score_frame.max(axis=1)
    confident_mask = top_conf >= threshold

    confident_count = int(confident_mask.sum())
    total_count = int(len(detected_df))
    uncertain_count = total_count - confident_count

    if confident_count > 0:
        dominant = top_label[confident_mask].value_counts(normalize=True)
    else:
        dominant = pd.Series(dtype=float)

    lines: List[str] = [
        "## 可讀摘要",
        "",
        f"- 判讀門檻（主行為信心度）：**{threshold:.0%}**",
        f"- 高於門檻的幀數：**{confident_count}/{total_count}**（{confident_count / total_count:.1%}）",
        f"- 不確定幀數：**{uncertain_count}/{total_count}**（{uncertain_count / total_count:.1%}）",
    ]

    if confident_count == 0:
        lines += [
            "",
            "目前所有幀都低於判讀門檻，建議將結果視為 **不確定**，回看原影片再判讀。",
        ]
    else:
        lines += ["", "### 主行為占比（僅計入高於門檻幀）"]
        for behavior, ratio in dominant.items():
            lines.append(f"- **{behavior}**：{ratio:.1%}")

    # Per-second view: pick the highest-confidence frame in each second.
    per_sec = pd.DataFrame(
        {
            "second": detected_df["timestamp"].fillna(0).astype(float).round().astype(int),
            "behavior": top_label,
            "confidence": top_conf,
            "is_confident": confident_mask,
        }
    )
    per_sec = per_sec.sort_values(["second", "confidence"], ascending=[True, False])
    per_sec = per_sec.drop_duplicates(subset=["second"], keep="first")

    lines += ["", "### 每秒主行為（前 20 秒）", "", "| 秒 | 主行為 | 信心度 | 狀態 |", "|---:|---|---:|---|"]
    for _, row in per_sec.head(max_rows).iterrows():
        status = "可判讀" if bool(row["is_confident"]) else "不確定"
        lines.append(
            f"| {int(row['second'])} | {row['behavior']} | {float(row['confidence']):.2%} | {status} |"
        )

    return "\n".join(lines)


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
    prompt_mode: str,
    prompt_llm_model: str,
    openrouter_api_key: str,
    prompt_aggregate: str,
    context_mode: str,
    sequence_aggregate: str,
    sequence_window: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str], str, str]:
    """Run the full pet-behaviour pipeline and return UI outputs.

    Returns
    -------
    readable_summary_md, summary_json, report_md, timeline_png, heatmap_png, dist_png, csv_path, prompt_json
    """
    if video_file is None:
        return (
            "無可讀摘要 – 請上傳影片後重新分析",
            "錯誤：請先上傳影片檔案",
            "無報告 – 請上傳影片後重新分析",
            None,
            None,
            None,
            "",
            "{}",
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

    # 2 – Prompt generation + classification
    progress(0.2, desc="SigLIP 零樣本分類中 …")
    prompt_result = build_label_prompt_result(
        labels=label_list,
        mode=prompt_mode,
        llm_model=prompt_llm_model,
        llm_api_key=openrouter_api_key.strip() or None,
    )
    prompt_map = prompt_result["prompt_map"]
    prompt_json = json.dumps(prompt_result, indent=2, ensure_ascii=False)
    prompt_list = flatten_prompt_map(prompt_map)

    classifier = SigLIPClassifier(model_name=model_name)

    if context_mode == "daynight":
        day_map = add_context_suffix(prompt_map, "captured during daytime")
        night_map = add_context_suffix(prompt_map, "captured during nighttime")

        day_prompts = flatten_prompt_map(day_map)
        night_prompts = flatten_prompt_map(night_map)

        day_prompt_scores_df = classifier.classify_frames(frames, day_prompts, timestamps)
        night_prompt_scores_df = classifier.classify_frames(frames, night_prompts, timestamps)

        day_scores_df = aggregate_prompt_scores(day_prompt_scores_df, day_map, reducer=prompt_aggregate)
        night_scores_df = aggregate_prompt_scores(night_prompt_scores_df, night_map, reducer=prompt_aggregate)

        p_night = estimate_night_probability(frames)
        scores_df = mix_day_night_scores(day_scores_df, night_scores_df, p_night)
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

    # 3 – Smoothing
    progress(0.5, desc="時序平滑處理 …")
    smoothed_df = smooth_scores(scores_df, window=smooth_window, method=smooth_method)

    # 4 – Anomaly detection
    progress(0.65, desc="異常偵測中 …")
    detector = AnomalyDetector(method=anomaly_method, threshold=threshold)
    detected_df = detector.detect(smoothed_df)
    summary_dict = detector.summary(detected_df)
    summary_dict["context_mode"] = context_mode
    summary_dict["sequence_aggregate"] = sequence_aggregate
    summary_dict["sequence_window"] = sequence_window
    summary_dict["ece"] = compute_ece_from_labeled_scores(detected_df)
    summary_dict["prompt_source"] = prompt_result.get("source")
    summary_dict["prompt_fallback_used"] = bool(prompt_result.get("fallback_used", False))
    summary_dict["prompt_fallback_reason"] = prompt_result.get("fallback_reason")

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
        llm_api_key=openrouter_api_key.strip() or None,
    )

    readable_summary = _build_readable_summary(detected_df, label_list)
    if bool(prompt_result.get("fallback_used", False)):
        reason = str(prompt_result.get("fallback_reason") or "未知原因")
        readable_summary += (
            "\n\n### Prompt 生成狀態\n"
            "- 偵測到 LLM prompt 生成失敗，已自動 fallback 到 template。\n"
            f"- 原因：`{reason}`"
        )

    progress(1.0, desc="完成！")
    summary_text = json.dumps(summary_dict, indent=2, ensure_ascii=False)
    return (
        readable_summary,
        summary_text,
        report_md,
        str(tl_path),
        str(hm_path),
        str(dist_path),
        str(csv_path),
        prompt_json,
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
                    label="報告模式（llm 需要 OPENROUTER_API_KEY）",
                )
                with gr.Row():
                    prompt_mode = gr.Dropdown(
                        choices=["off", "template", "llm"],
                        value="template",
                        label="Prompt 生成模式",
                    )
                    prompt_aggregate = gr.Dropdown(
                        choices=["max", "mean"],
                        value="max",
                        label="Prompt 分數聚合",
                    )
                prompt_llm_model = gr.Textbox(
                    label="Prompt LLM 模型（prompt_mode=llm 時使用）",
                    value="openai/gpt-4o-mini",
                )
                openrouter_api_key = gr.Textbox(
                    label="OpenRouter API Key（可直接輸入，不用環境變數）",
                    value="",
                    type="password",
                )
                with gr.Row():
                    context_mode = gr.Dropdown(
                        choices=["off", "daynight"],
                        value="off",
                        label="情境模式（day/night）",
                    )
                    sequence_aggregate = gr.Dropdown(
                        choices=["none", "prob", "logit"],
                        value="none",
                        label="序列聚合模式",
                    )
                sequence_window = gr.Slider(
                    minimum=3,
                    maximum=21,
                    step=2,
                    value=5,
                    label="序列聚合視窗",
                )
                run_btn = gr.Button("🚀 開始分析", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 分析結果")
                with gr.Tabs():
                    with gr.Tab("摘要 & 報告"):
                        readable_summary_out = gr.Markdown(label="可讀摘要")
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
                    with gr.Tab("Prompt 檢視"):
                        prompt_out = gr.Code(label="實際使用的 Prompt Map (JSON)", language="json")

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
                prompt_mode,
                prompt_llm_model,
                openrouter_api_key,
                prompt_aggregate,
                context_mode,
                sequence_aggregate,
                sequence_window,
            ],
            outputs=[
                readable_summary_out,
                summary_out,
                report_out,
                timeline_out,
                heatmap_out,
                dist_out,
                csv_out,
                prompt_out,
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
