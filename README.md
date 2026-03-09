# pet-behavior-clip

Local pet behavior analysis using SigLIP zero-shot classification.

## Pipeline

```text
Input video
  -> frame sampling (video.py)
  -> SigLIP frame scoring (clip_zeroshot.py)
  -> optional sequence aggregation (contextual.py)
  -> temporal smoothing (smoothing.py)
  -> anomaly detection (anomaly.py)
  -> CSV/JSON/plots/local markdown report
```

This project is zero-cloud by design:
- No external API calls
- No LLM prompt expansion
- No environment-variable API key checks

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## CLI

```bash
pet-behavior-clip analyze my_dog.mp4

pet-behavior-clip analyze my_dog.mp4 \
  --labels "a picture of an animal moving,a picture of an animal eating,a picture of an animal resting" \
  --fps 2 \
  --smooth-window 7 \
  --smooth-method rolling_mean \
  --anomaly-method zscore \
  --threshold 2.5 \
  --sequence-aggregate prob \
  --sequence-window 7 \
  --output-dir ./results
```

## UI

```bash
python ui/app.py
```

Then open `http://localhost:7860`.

## Python API

```python
from pet_behavior_clip.video import VideoReader
from pet_behavior_clip.clip_zeroshot import SigLIPClassifier
from pet_behavior_clip.contextual import aggregate_sequence_scores
from pet_behavior_clip.smoothing import smooth_scores
from pet_behavior_clip.anomaly import AnomalyDetector
from pet_behavior_clip.report import generate_report

labels = [
    "a picture of an animal moving",
    "a picture of an animal eating",
    "a picture of an animal resting",
]

reader = VideoReader("my_dog.mp4", sample_fps=1.0)
frame_data = reader.sample_frames()
reader.release()

timestamps = [t for t, _ in frame_data]
frames = [img for _, img in frame_data]

clf = SigLIPClassifier()
scores = clf.classify_frames(frames, labels, timestamps)

scores = aggregate_sequence_scores(scores, mode="prob", window=5)
smoothed = smooth_scores(scores, window=5, method="rolling_mean")

detector = AnomalyDetector(method="zscore", threshold=2.5)
detected = detector.detect(smoothed)
summary = detector.summary(detected)

report = generate_report(detected, labels, video_path="my_dog.mp4")
print(summary)
print(report)
```

## Output Files

- `*_scores.csv`: per-frame scores + anomaly fields
- `*_summary.json`: aggregate anomaly summary
- `*_timeline.png`: behavior timeline
- `*_heatmap.png`: behavior heatmap
- `*_distribution.png`: score distribution
- `*_report.md`: local deterministic report

## Tests

```bash
pytest tests -v
```
