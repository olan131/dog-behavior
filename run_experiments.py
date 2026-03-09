import time
from pathlib import Path
import pandas as pd
# 確保引用路徑正確，假設您的資料夾名稱為 pet_behavior_clip
from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.smoothing import smooth_scores
from pet_behavior_clip.plots import plot_behavior_timeline

def calculate_label_flip_rate(df):
    """計算標籤跳變率：衡量時序穩定性的核心指標"""
    # 排除 metadata 欄位
    score_cols = [c for c in df.columns if c not in ["timestamp", "anomaly_score", "is_anomaly"]]
    # 取得每一幀機率最高的標籤
    predictions = df[score_cols].idxmax(axis=1)
    # 計算相鄰幀標籤不同的次數
    flips = (predictions != predictions.shift()).sum() - 1  # 扣除第一幀的位移
    return (flips / (len(df) - 1)) * 100 if len(df) > 1 else 0

def run_experiment_suite(video_path, labels, output_dir="experiment_results"):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    chunk_size = 64
    
    print(f"[*] Loading model and video: {video_path}")
    classifier = SigLIPClassifier()
    
    # 實驗 B: 效能基準測試
    with VideoReader(video_path, sample_fps=2.0) as reader:
        print("[*] Running Inference (Experiment B)...")
        start_time = time.time()
        parts = []
        batch_timestamps = []
        batch_images = []
        sampled_count = 0

        # Stream frame sampling + inference to reduce startup latency and memory.
        for timestamp, image in reader.iter_frames():
            batch_timestamps.append(timestamp)
            batch_images.append(image)

            if len(batch_images) >= chunk_size:
                part = classifier.classify_frames(
                    batch_images,
                    labels,
                    timestamps=batch_timestamps,
                )
                parts.append(part)
                sampled_count += len(batch_images)
                print(f"    processed {sampled_count} sampled frames...")
                batch_timestamps = []
                batch_images = []

        if batch_images:
            part = classifier.classify_frames(
                batch_images,
                labels,
                timestamps=batch_timestamps,
            )
            parts.append(part)
            sampled_count += len(batch_images)

        if not parts:
            print("Error: No frames found in video.")
            return

        raw_df = pd.concat(parts, ignore_index=True)
        
        total_time = time.time() - start_time
        fps = len(raw_df) / total_time
        
    print(f"\n[Experiment B Result]")
    print(f"- Inference FPS: {fps:.2f}")

    # 實驗 A: 時序平滑穩定性分析
    print(f"\n[*] Running Stability Analysis (Experiment A)...")
    
    # 1. 原始數據
    raw_flip = calculate_label_flip_rate(raw_df)
    plot_behavior_timeline(raw_df, output_path=out_path / "exp_A_raw.png")
    
    # 2. 平滑後數據 (Window=15)
    smooth_df = smooth_scores(raw_df, window=15, method="gaussian")
    smooth_flip = calculate_label_flip_rate(smooth_df)
    plot_behavior_timeline(smooth_df, output_path=out_path / "exp_A_smoothed.png")
    
    print(f"- Raw Label Flip Rate: {raw_flip:.2f}%")
    print(f"- Smoothed Label Flip Rate: {smooth_flip:.2f}%")
    print(f"- Stability Improvement: {raw_flip - smooth_flip:.2f}%")

    # 保存數據
    raw_df.to_csv(out_path / "data_raw.csv", index=False)
    smooth_df.to_csv(out_path / "data_smooth.csv", index=False)

if __name__ == "__main__":
    # 使用原始字串 (r"...") 解決 Windows 路徑轉義問題
    TEST_VIDEO = r"D:\\porject\\dog2.mp4"
    if not Path(TEST_VIDEO).exists():
        print(f"Error: video path does not exist: {TEST_VIDEO}")
        raise SystemExit(1)
    TEST_LABELS = [
        "a picture of a dog sleeping",
        "a picture of a dog walking",
        "a picture of a dog eating"
    ]
    
    run_experiment_suite(TEST_VIDEO, TEST_LABELS)