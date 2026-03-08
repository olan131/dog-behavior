# pet-behavior-clip 🐾
test
> 基於 SigLIP 零樣本分類的寵物行為異常偵測系統  
> Pet behaviour anomaly detection via SigLIP / CLIP zero-shot classification

---

## 專案概述

**pet-behavior-clip** 是一個以電腦視覺為核心的 Python 專案，透過 Google SigLIP（Sigmoid Loss for Language-Image Pre-Training）模型，對寵物影片進行零樣本行為分類，並自動標記異常行為片段。

### 應用場景
- 寵物健康監控（偵測跛行、過度抓癢、異常顫抖等）
- 動物行為研究
- 智慧寵物攝影機後端分析

---

## 架構與資料流程

```
影片 (.mp4/.avi/…)
    │
    ▼
[video.py]  VideoReader
    │  以指定 FPS 取樣影像幀
    ▼
[clip_zeroshot.py]  SigLIPClassifier
    │  零樣本分類 → DataFrame (timestamp × labels)
    ▼
[smoothing.py]  smooth_scores
    │  時序平滑 (rolling mean / Gaussian / EWM)
    ▼
[anomaly.py]  AnomalyDetector
    │  Z-score / IQR 異常偵測 → is_anomaly 旗標
    ▼
[plots.py]  視覺化
    │  時間線圖 / 熱力圖 / 分佈圖 → ui_output/
    ▼
[report_llm.py]  generate_report
       模板或 GPT-4o-mini 報告 → Markdown
```

---

## 模組說明

| 模組 | 功能 |
|------|------|
| `pet_behavior_clip/video.py` | 使用 OpenCV 讀取影片，依指定 fps 取樣幀，回傳 `(timestamp, PIL.Image)` 列表 |
| `pet_behavior_clip/clip_zeroshot.py` | 載入 SigLIP / CLIP 模型，批次推理，回傳每幀對每個行為標籤的信心分數 DataFrame |
| `pet_behavior_clip/smoothing.py` | 對分數時序資料套用滑動平均、Gaussian 卷積或指數加權平均，降低單幀雜訊 |
| `pet_behavior_clip/anomaly.py` | Z-score 或 IQR 方法標記偏離基準的異常幀，附帶摘要統計 |
| `pet_behavior_clip/plots.py` | 繪製行為時間線、熱力圖、信心度分佈圖，並可儲存為 PNG |
| `pet_behavior_clip/report_llm.py` | 依統計摘要產生繁體中文分析報告；支援模板模式與 OpenRouter LLM 模式 |
| `pet_behavior_clip/cli.py` | Click-based CLI，一行指令完成完整分析流程 |
| `ui/app.py` | Gradio 互動式網頁 UI |

---

## 快速開始

### 安裝

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

### CLI 分析

```bash
# 使用預設行為標籤分析影片
pet-behavior-clip analyze my_dog.mp4

# 自訂標籤、輸出目錄與參數
pet-behavior-clip analyze my_dog.mp4 \
    --labels "a picture of an animal moving,a picture of an animal eating,a picture of an animal resting" \
    --fps 2 \
    --smooth-window 7 \
    --anomaly-method zscore \
    --threshold 2.5 \
    --output-dir ./results

# 使用 LLM 生成報告（需設定 OPENROUTER_API_KEY）
OPENROUTER_API_KEY=sk-or-... pet-behavior-clip analyze my_dog.mp4 --report-mode llm

# 或直接用指令帶 key（不依賴環境變數）
pet-behavior-clip analyze my_dog.mp4 --report-mode llm --openrouter-api-key "sk-or-..."

# 啟用 LLM-augmented prompt 生成（將類別擴充為多個語義提示）
OPENROUTER_API_KEY=sk-or-... pet-behavior-clip analyze my_dog.mp4 \
    --labels "Active,Resting,Eating/Drinking" \
    --prompt-mode llm \
    --prompt-aggregate max

# 直接帶 key 的 prompt 生成
pet-behavior-clip analyze my_dog.mp4 \
    --labels "Active,Resting,Eating/Drinking" \
    --prompt-mode llm \
    --prompt-aggregate max \
    --openrouter-api-key "sk-or-..."

# 僅使用本地模板 prompt 擴充（不呼叫 API）
pet-behavior-clip analyze my_dog.mp4 \
    --labels "Active,Resting,Eating/Drinking" \
    --prompt-mode template

# 啟用 day/night 情境混合 + 序列聚合
pet-behavior-clip analyze my_dog.mp4 \
    --labels "Active,Resting,Eating/Drinking" \
    --context-mode daynight \
    --sequence-aggregate logit \
    --sequence-window 7
```

### Gradio 網頁 UI

```bash
python ui/app.py
# 開啟 http://localhost:7860

UI 中可透過「Prompt 生成模式」選擇：
- `off`：直接使用原始 labels
- `template`：使用本地模板擴充 prompts
- `llm`：使用 OpenRouter 生成 prompts（需 `OPENROUTER_API_KEY`）

另可設定：
- `情境模式（day/night）`：以畫面亮度估計夜間機率，混合 daytime/nighttime prompts 分數
- `序列聚合模式`：`none` / `prob` / `logit`，可降低單幀抖動
```

### Python API

```python
from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.smoothing import smooth_scores
from pet_behavior_clip.anomaly import AnomalyDetector
from pet_behavior_clip.plots import plot_behavior_timeline
from pet_behavior_clip.report_llm import generate_report

labels = [
    "a picture of an animal moving",
    "a picture of an animal eating",
    "a picture of an animal resting",
]

# 1. 取樣影片幀
reader = VideoReader("my_dog.mp4", sample_fps=1.0)
frame_data = reader.sample_frames()
timestamps, frames = zip(*frame_data)

# 2. 零樣本分類
clf = SigLIPClassifier()
scores = clf.classify_frames(list(frames), labels, list(timestamps))

# 3. 時序平滑
smoothed = smooth_scores(scores, window=5)

# 4. 異常偵測
detector = AnomalyDetector(method="zscore", threshold=2.5)
detected = detector.detect(smoothed)
print(detector.summary(detected))

# 5. 視覺化 & 報告
plot_behavior_timeline(detected, output_path="timeline.png")
report = generate_report(detected, labels, mode="template")
print(report)
```

---

## 輸出說明 (`ui_output/`)

| 檔案 | 說明 |
|------|------|
| `*_scores.csv` | 每幀時間戳記、各行為信心分數、異常分數與旗標 |
| `*_summary.json` | 偵測摘要（總幀數、異常幀數、最嚴重時間點等） |
| `*_timeline.png` | 行為信心度隨時間變化折線圖（異常區間以紅色標示） |
| `*_heatmap.png` | 行為 × 時間熱力圖 |
| `*_distribution.png` | 各行為信心度箱形圖 |
| `*_report.md` | Markdown 格式分析報告（繁體中文） |

---

## 執行測試

```bash
pip install pytest
pytest tests/ -v
```

---

## 核心技術

- **SigLIP / CLIP**: 零樣本視覺-語言對齊模型，無需行為標注資料即可分類
- **pandas / numpy**: 時序資料處理與統計分析
- **matplotlib**: 科學視覺化
- **OpenRouter + LLM Provider** (選用): 自然語言報告生成
- **Gradio**: 快速建立互動式 ML 展示 UI
- **Click**: 強型別 CLI

---

## 論文參考架構

若使用本專案支援學術研究，建議引用以下架構段落：

### 方法章節摘要
1. **影像取樣**：以固定頻率（預設 1 fps）從影片取樣幀，保留時間戳記。
2. **零樣本分類**：將幀與行為描述文字送入 SigLIP，計算跨模態相似度（sigmoid 機率）。
3. **時序平滑**：滑動平均（或 Gaussian）降低單幀噪音，提升序列一致性。
4. **異常偵測**：基於 Z-score 計算各幀偏離基準的程度，超過閾值者標記為異常。
5. **輸出生成**：產生 CSV、JSON 摘要、視覺化圖表與 Markdown 報告。

---

## 授權

MIT License
