# pet-behavior-clip 專案審查報告

## 專案概覽

本專案使用 SigLIP 零樣本分類模型，對寵物影片進行行為分析與異常偵測。整體架構清晰、模組化程度佳。以下為審查後發現的潛在問題，依嚴重程度分類。

---

## 🔴 嚴重問題

### 1. [video.py](file:///d:/dog-behavior/tests/test_video.py) — 使用非 `grab()` 路徑時可能存取 `None`

[video.py:L117-121](file:/// d:/dog-behavior/pet_behavior_clip/video.py#L117-L121)

當 `has_grab = False`（理論上不太常見但仍為合法路徑）且 `frame_idx % step != 0` 時，`bgr` 已被讀取但不會進入 `yield`，這沒問題。但在 `has_grab = True` 的路徑中，若 `frame_idx % step != 0`，`bgr` 仍為 `None`，程式正常跳過不會出錯。然而若 `cap.grab()` 成功但 `cap.retrieve()` 失敗，程式 break 但此時 `bgr` 仍是 `None`，在後續的 `cvtColor` 前已經 break，所以實際上不會出事。**但整體流程較脆弱，建議重構讓意圖更清晰。**

### 2. [clip_zeroshot.py](file:///d:/dog-behavior/tests/test_clip_zeroshot.py) — Image / Text 特徵向量未做 L2 正規化

[clip_zeroshot.py:L227-237](file:///d:/dog-behavior/pet_behavior_clip/clip_zeroshot.py#L227-L237)

CLIP / SigLIP 的標準作法是在計算 cosine similarity 前對 `image_features` 和 `text_features` 做 L2 正規化。目前程式碼**未做正規化就直接相乘**，依賴 `logit_scale` 和 `logit_bias` 彌補。在 SigLIP 官方實作中，`get_image_features()` / `get_text_features()` 的輸出**不保證已正規化**，這可能導致：
- 相似度分數不在預期範圍
- 不同長度的 text embedding 產生不公平的比較

```diff
+ # 建議加入 L2 正規化
+ image_features = image_features / image_features.norm(dim=-1, keepdim=True)
+ text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

### 3. [anomaly.py](file:///d:/dog-behavior/tests/test_anomaly.py) — IQR 方法未使用使用者設定的 `threshold`

[anomaly.py:L163-164](file:///d:/dog-behavior/pet_behavior_clip/anomaly.py#L163-L164)

[_iqr_deviations](file:///d:/dog-behavior/pet_behavior_clip/anomaly.py#156-168) 內部**硬編碼 `1.5`** 作為 IQR 倍數，而非使用 `self.threshold`：

```python
lower = q1 - 1.5 * iqr  # 硬編碼 1.5，忽略使用者設定的 threshold
upper = q3 + 1.5 * iqr
```

使用者透過 CLI 設定的 `--threshold` 只被用來做最終 `anomaly_score > threshold` 的判斷，但 IQR 的邊界計算始終用 1.5。這意味著 IQR 方法的靈敏度無法被使用者調整。

---

## 🟡 中等問題

### 4. [setup.py](file:///d:/dog-behavior/setup.py) — `python_requires=">=3.9"` 但程式碼使用 `dict[str, list[str]]` 語法

[prompt.py:L110](file:///d:/dog-behavior/pet_behavior_clip/prompt.py#L110)、[plots.py:L313](file:///d:/dog-behavior/pet_behavior_clip/plots.py#L313)

多處使用 Python 3.10+ 的 `dict[str, list[str]]` 內建泛型語法，但 [setup.py](file:///d:/dog-behavior/setup.py) 聲明 `python_requires=">=3.9"`。在 Python 3.9 中，這些語法會在**執行期**拋 `TypeError`，因為 `from __future__ import annotations` 只延遲了**註解求值**，但函式簽名中的型別提示在某些框架下仍可能被求值。

> [!IMPORTANT]
> 建議統一為 `Dict[str, List[str]]`（從 `typing` 匯入），或將 `python_requires` 改為 `">=3.10"`。

### 5. [prompt.py](file:///d:/dog-behavior/tests/test_prompt.py) — 不支援自訂 labels，硬編碼僅支援 6 種行為

[prompt.py:L197-204](file:///d:/dog-behavior/pet_behavior_clip/prompt.py#L197-L204)

[_template_prompts](file:///d:/dog-behavior/pet_behavior_clip/prompt.py#197-205) 遇到不在 `_CUSTOM_PROMPTS` 中的 label 會直接拋 `ValueError`。README 和 CLI 都暗示使用者可自訂 labels，但實際上 [classify_with_template_max](file:///d:/dog-behavior/pet_behavior_clip/prompt.py#159-176) 只支援 6 種預定義行為。**使用者若輸入 `"barking"` 等自訂 label，程式會直接崩潰。**

建議：為未知 label 回退到 `f"a picture of a dog {label}"` 的通用 prompt。

### 6. 記憶體使用 — [sample_frames()](file:///d:/dog-behavior/pet_behavior_clip/video.py#69-75) 一次性載入所有幀到記憶體

[video.py:L69-74](file:///d:/dog-behavior/pet_behavior_clip/video.py#L69-L74) 及 [cli.py:L137-140](file:///d:/dog-behavior/pet_behavior_clip/cli.py#L137-L140)

CLI 和 UI 都使用 `reader.sample_frames()`（一次產生所有幀的 list），而非 [iter_frames()](file:///d:/dog-behavior/pet_behavior_clip/video.py#76-82) 的串流模式。對於長影片（例如數小時的監視器錄影），這可能消耗數 GB 記憶體。雖然 [iter_frames()](file:///d:/dog-behavior/pet_behavior_clip/video.py#76-82) 已存在，但管線並未利用它。

### 7. [smoothing.py](file:///d:/dog-behavior/tests/test_smoothing.py) — Gaussian kernel 長度與 window 不匹配

[smoothing.py:L96-101](file:///d:/dog-behavior/pet_behavior_clip/smoothing.py#L96-L101)

[_gaussian_kernel(window, sigma)](file:///d:/dog-behavior/pet_behavior_clip/smoothing.py#96-102) 產生的 kernel 長度為 `2*(window//2)+1`。當 `window` 為偶數時（例如 `window=4`），實際 kernel 長度是 `2*2+1=5`，**與使用者指定的 window 大小不一致**。CLI 限制了 step=2 所以目前不會出問題，但 Python API 使用者可能傳入偶數值。

### 8. [plots.py](file:///d:/dog-behavior/tests/test_plots.py) — matplotlib Figure 未被正確關閉

所有 plot 函式回傳 `fig` 物件但**從未呼叫 `plt.close(fig)`**。在 UI 或批次處理中反覆呼叫時，matplotlib 會累積 Figure 物件導致**記憶體洩漏**。

```diff
  if output_path:
      _save(fig, output_path)
+ plt.close(fig)
  return fig
```

### 9. [behavior_postprocess.py](file:///d:/dog-behavior/tests/test_behavior_postprocess.py) — `second_scores` 計算效率極差

[behavior_postprocess.py:L41](file:///d:/dog-behavior/pet_behavior_clip/behavior_postprocess.py#L41)

```python
second_scores = score_df.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
```

`apply` + `nlargest` 是逐行 Python 迴圈，效能非常差。在大型 DataFrame 上會成為瓶頸。建議改用 numpy：

```python
vals = score_df.to_numpy()
sorted_vals = np.sort(vals, axis=1)
second_scores = pd.Series(sorted_vals[:, -2], index=score_df.index)
```

---

## 🟢 輕微問題 / 改善建議

### 10. [.gitignore](file:///d:/dog-behavior/.gitignore) — [.mp4](file:///d:/dog-behavior/dog.mp4) 規則寫法錯誤

[.gitignore:L31](file:///d:/dog-behavior/.gitignore#L31)

```
.mp4
```

這只會忽略名為 [.mp4](file:///d:/dog-behavior/dog.mp4) 的檔案（隱藏檔），而非 `*.mp4`（所有 mp4 檔案）。實際上 [dog.mp4](file:///d:/dog-behavior/dog.mp4) 已被追蹤到 repo 中（58 MB），這會拖慢 git 操作。

### 11. [setup.py](file:///d:/dog-behavior/setup.py) — `gradio` 不應列為核心依賴

[setup.py:L26](file:///d:/dog-behavior/setup.py#L26)

`gradio>=4.0.0` 被列在 `install_requires` 中，但它只有 UI 才需要。建議改為 `extras_require`：

```python
extras_require={
    "ui": ["gradio>=4.0.0"],
},
```

### 12. [cli.py](file:///d:/dog-behavior/pet_behavior_clip/cli.py) — entry point 名稱 [main](file:///d:/dog-behavior/pet_behavior_clip/cli.py#240-242) 指向 [cli()](file:///d:/dog-behavior/pet_behavior_clip/cli.py#31-35) 非 [main()](file:///d:/dog-behavior/pet_behavior_clip/cli.py#240-242)

[cli.py:L30-31](file:///d:/dog-behavior/pet_behavior_clip/cli.py#L30-L31) 及 [setup.py:L29-32](file:///d:/dog-behavior/setup.py#L29-L32)

[setup.py](file:///d:/dog-behavior/setup.py) 定義了兩個 console_scripts entry point：
- `pet-behavior-clip=pet_behavior_clip.cli:main` ✅ 正確
- `pet-behavior-ui=ui.app:main` — [ui](file:///d:/dog-behavior/ui/app.py#204-330) 資料夾被 `find_packages(exclude=["tests*", "ui"])` 排除了，所以**透過 `pip install -e .` 安裝後此 entry point 會找不到 `ui.app` 模組**。

### 13. [contextual.py](file:///d:/dog-behavior/tests/test_contextual.py) — 缺少函式間的空行

[contextual.py:L8-9](file:///d:/dog-behavior/pet_behavior_clip/contextual.py#L8-L9)

`import` 和 `def` 之間缺少 PEP 8 規範的兩個空行。

### 14. [report.py](file:///d:/dog-behavior/tests/test_report.py) — [duration](file:///d:/dog-behavior/pet_behavior_clip/video.py#59-64) 僅取最後一幀的 timestamp，非影片實際長度

[report.py:L27-28](file:///d:/dog-behavior/pet_behavior_clip/report.py#L27-L28)

[duration](file:///d:/dog-behavior/pet_behavior_clip/video.py#59-64) 取的是最後一幀的 timestamp，不是影片的實際長度。若 `sample_fps` 設很低，最後一幀的 timestamp 可能遠小於影片實際長度。

### 15. 缺少型別檢查 / Linting 設定

專案沒有 `pyproject.toml`、`mypy.ini`、`ruff.toml` 等設定檔。建議加入 `ruff` 和 `mypy` 設定以確保程式碼品質。

---

## 總結

| 類別 | 數量 |
|---|---|
| 🔴 嚴重問題 | 3 |
| 🟡 中等問題 | 6 |
| 🟢 輕微 / 建議 | 6 |

專案整體架構設計良好，模組劃分清楚，lazy import 和 context manager 等設計都很正確。主要風險集中在：
1. **特徵向量未正規化**可能影響分類品質
2. **自訂 label 直接崩潰**影響使用者體驗
3. **IQR 硬編碼**導致使用者設定無效
