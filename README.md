# pet-behavior-clip（現況整理）

這個專案目前同時包含兩類內容：

1. **主專案功能**：`pet_behavior_clip/`、`ui/`、CLI、可安裝套件。
2. **研究/實驗腳本**：`ablation1.py`、`ablation2.py`、`ablation3.py`、`manual_label.py`、`benchmark_inference.py`。

你目前看到的「混在一起」是事實：同一個 repo 既有產品化流程，也有論文/消融與人工標註工具。  
本 README 先不改程式碼，只做完整盤點：**目前有哪些功能、各自怎麼做**。

---

## 1) 主專案：目前完整功能與做法

### 1.1 端到端分析流程（CLI/UI 共用）

```text
輸入影片
  -> 取樣影格 (video.py)
  -> SigLIP 零樣本分類 (clip_zeroshot.py + prompt.py)
  -> 時間平滑 (smoothing.py)
  -> 異常偵測 (anomaly.py)
  -> 後處理行為標籤/區段 (behavior_postprocess.py, UI 會用)
  -> 輸出 CSV / JSON / 圖 / Markdown 報告
```

### 1.2 `video.py`：影片讀取與抽幀

- 功能：以 OpenCV 讀影片，輸出 `(timestamp, PIL.Image)`。
- 做法：
  - `VideoReader(path, sample_fps)` 建立讀取器。
  - 依 `native_fps / sample_fps` 計算步長抽樣。
  - 使用 `grab/retrieve`（可用時）降低不必要解碼成本。
  - 支援 `sample_frames()`（一次回傳全部）與 `iter_frames()`（生成器）。

### 1.3 `clip_zeroshot.py`：SigLIP 零樣本分類

- 功能：把每張影格對多個文字提示做相似度推論，輸出每類別分數（DataFrame）。
- 做法：
  - 預設模型：`google/siglip-so400m-patch14-224`，失敗時回退 `openai/clip-vit-base-patch32`。
  - 文字嵌入做快取（同一組 prompt 不重編碼）。
  - 影像分批推論（`batch_size`），最後 softmax 成每幀機率分數。

### 1.4 `prompt.py`：提示詞策略（目前核心差異來源）

- 功能：建立每個行為標籤的多提示詞變體，並把 prompt 分數聚合成「每類一欄」。
- 做法：
  - 每個 label 提供 5 個描述（`_CUSTOM_PROMPTS`，例如 running/eating/walking...）。
  - `classify_with_template_max()`：多提示詞 + `max` 聚合（目前系統主路徑）。
  - `classify_with_single_prompt()`：每類只取第一個提示詞（Single Prompt 基線）。
  - `aggregate_prompt_scores()`：`max` 或 `mean` 聚合後再做逐列正規化。

### 1.5 `smoothing.py`：時間平滑

- 功能：降低幀間抖動，讓時間序列更穩定。
- 做法：
  - `rolling_mean`：滑動平均。
  - `gaussian`：高斯 kernel 卷積。
  - `exponential`：指數加權移動平均（EWM）。

### 1.6 `anomaly.py`：異常偵測

- 功能：標記異常幀，新增 `anomaly_score` 與 `is_anomaly`。
- 做法：
  - `zscore`：看各類分數偏離均值的標準差幅度。
  - `iqr`：用四分位距上下界衡量離群程度。
  - `summary()` 回傳總幀數、異常比例、最嚴重時間點等摘要。

### 1.7 `behavior_postprocess.py`：行為標籤後處理（UI 主要使用）

- 功能：從分數推導可讀行為時間線。
- 做法：
  - `infer_frame_behaviors()`：
    - 先取 top-1 類別；
    - 再檢查信心門檻與 top1-top2 margin；
    - 異常幀優先標 `anomaly`，其餘不確定標 `uncertain`。
  - `smooth_behavior_labels()`：時間窗多數決，降低標籤閃爍。
  - `build_behavior_segments()`：把連續相同標籤合併成區段。
  - `summarize_behavior_results()`：輸出可讀 Markdown 摘要。

### 1.8 `plots.py`：視覺化

- `plot_behavior_timeline`：各類分數隨時間折線圖（可疊異常區間）。
- `plot_anomaly_heatmap`：類別 x 時間熱圖（可標異常時間線）。
- `plot_confidence_distribution`：各類分數分布圖（box plot）。
- `plot_behavior_segments_timeline`：後處理後的行為區段時間線。

### 1.9 `report.py` + `contextual.py`：文字報告與校準指標

- `generate_report()`：輸出本地 Markdown 報告（幀數、異常比例、各類統計、峰值異常）。
- `compute_ece_from_labeled_scores()`：
  - 若資料含 `gt_label`，可計算 Expected Calibration Error (ECE)；
  - 無標註欄位則回傳 `None`。

---

## 2) 使用介面

### 2.1 CLI（`pet_behavior_clip/cli.py`）

指令：

```bash
pet-behavior-clip analyze my_dog.mp4
```

可調參數（重點）：

- `--labels`：分類標籤清單（逗號分隔）
- `--fps`：抽幀率
- `--smooth-window`、`--smooth-method`
- `--anomaly-method`、`--threshold`
- `--model`：HF 模型名稱
- `--output-dir`：輸出目錄

CLI 的實際步驟是固定 5 段：
1) 抽幀  
2) SigLIP + prompt 分類  
3) 分數平滑  
4) 異常偵測 + 摘要  
5) 存 CSV/JSON/圖/MD 報告

### 2.2 Web UI（`ui/app.py`）

啟動：

```bash
python ui/app.py
```

目前 UI 功能比 CLI 更完整，除基本流程外還包含：

- Prompt 模式切換：
  - Multi-prompt（5 variants + max）
  - Single prompt（D1 only）
- 行為後處理參數：
  - confidence threshold
  - margin threshold
  - label smoothing window
  - anomaly alert threshold
- 額外輸出：
  - `*_labels.csv`（每幀行為標籤）
  - `*_segments.csv`（區段化結果）
  - `*_behavior_timeline.png`（行為區段圖）

---

## 3) 實驗與研究腳本（非主流程）

### 3.1 `ablation1.py`：Single Prompt vs Multi-Prompt

- 目的：比較「單提示詞」和「多提示詞（max 聚合）」在各類別分數波動（std）。
- 固定標籤：`running,eating,walking,standing,sitting,lying`
- 輸出：`ablation1_prompt_mode.png`

### 3.2 `ablation2.py`：時間平滑消融（raw vs rolling_mean）

- 目的：只改平滑方法，其餘條件固定，比較可讀性/穩定性。
- 指標：
  - 主導類別切換次數（dominant switches）
  - 曲線交叉次數（pair crossings）
  - walking vs standing 交叉次數
  - top-1 margin
- 輸出：
  - `ablation2_fig2a_raw.png`
  - `ablation2_fig2b_rolling_mean_w5.png`
  - `ablation2_metrics.csv`
  - `ablation2_pair_crossings.csv`
  - `ablation2_summary.txt`

### 3.3 `ablation3.py`：提示詞配置準確率比較（依人工標註）

- 目的：用 `manual_labels.csv` 當 ground truth，比較多種 prompt 配置。
- 配置：
  - A: Single+max
  - B: Multi+max（目前系統）
  - C: Multi+mean
  - D: Baseline（`a dog [label]`）
- 輸出：
  - `ablation3_accuracy.png`
  - `ablation3_results.csv`
  - `ablation3_table.tex`

### 3.4 `manual_label.py`：人工快速標註/對照

- 目的：人工 spot-check，對比 System A（single）與 System B（multi）。
- 做法：
  - 開 OpenCV 視窗逐幀標註（1~6、s skip、q quit）。
  - 可 `random` 或 `stratified` 抽樣。
  - 產生 `manual_labels.csv` 與對照報告。

### 3.5 `benchmark_inference.py`：推論效能基準

- 比較三種策略：
  - A 無文字快取 + 單幀
  - B 文字快取 + 單幀
  - C 文字快取 + 批次
- 輸出：
  - `benchmark_results.png`
  - `benchmark_results.csv`

---

## 4) 輸出檔案（主流程常見）

- `*_scores.csv`：每幀各類分數 + 異常欄位
- `*_summary.json`：異常摘要與統計
- `*_timeline.png`：分數時間線
- `*_heatmap.png`：熱圖
- `*_distribution.png`：分數分布
- `*_report.md`：本地 Markdown 報告
- `*_labels.csv`：每幀行為標籤（UI 後處理）
- `*_segments.csv`：行為區段（UI 後處理）
- `*_behavior_timeline.png`：區段時間線（UI）

---

## 5) 安裝與測試

安裝：

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

測試：

```bash
pytest tests -v
```

---

## 6) 目前專案狀態（整理版結論）

- 這個 repo 現在是「**產品管線 + 實驗腳本 + 人工標註工具**」並存。
- `pet_behavior_clip/` 與 `ui/` 可視為主系統；`ablation*.py`、`manual_label.py`、`benchmark_inference.py` 屬研究評估工具。
- 功能上可運作，但文件與目錄語意目前確實混雜，後續若要更清楚，建議再做實體分層（例如 `src/`、`experiments/`、`tools/`）。
