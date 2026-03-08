"""pet-behavior-clip: Pet behavior anomaly detection via SigLIP zero-shot classification."""

__version__ = "0.1.0"
__author__ = "pet-behavior-clip contributors"

__all__ = [
    "VideoReader",
    "SigLIPClassifier",
    "AnomalyDetector",
    "smooth_scores",
    "plot_behavior_timeline",
    "plot_anomaly_heatmap",
    "plot_confidence_distribution",
    "generate_report",
    "build_label_prompt_map",
    "flatten_prompt_map",
    "aggregate_prompt_scores",
    "estimate_night_probability",
    "add_context_suffix",
    "mix_day_night_scores",
    "aggregate_sequence_scores",
    "compute_ece_from_labeled_scores",
]


def __getattr__(name: str):  # noqa: N807
    """Lazy imports – only load heavy modules when first accessed."""
    _lazy = {
        "VideoReader": ("pet_behavior_clip.video", "VideoReader"),
        "SigLIPClassifier": ("pet_behavior_clip.clip_zeroshot", "SigLIPClassifier"),
        "AnomalyDetector": ("pet_behavior_clip.anomaly", "AnomalyDetector"),
        "smooth_scores": ("pet_behavior_clip.smoothing", "smooth_scores"),
        "plot_behavior_timeline": ("pet_behavior_clip.plots", "plot_behavior_timeline"),
        "plot_anomaly_heatmap": ("pet_behavior_clip.plots", "plot_anomaly_heatmap"),
        "plot_confidence_distribution": (
            "pet_behavior_clip.plots",
            "plot_confidence_distribution",
        ),
        "generate_report": ("pet_behavior_clip.report_llm", "generate_report"),
        "build_label_prompt_map": ("pet_behavior_clip.prompt_llm", "build_label_prompt_map"),
        "flatten_prompt_map": ("pet_behavior_clip.prompt_llm", "flatten_prompt_map"),
        "aggregate_prompt_scores": ("pet_behavior_clip.prompt_llm", "aggregate_prompt_scores"),
        "estimate_night_probability": ("pet_behavior_clip.contextual", "estimate_night_probability"),
        "add_context_suffix": ("pet_behavior_clip.contextual", "add_context_suffix"),
        "mix_day_night_scores": ("pet_behavior_clip.contextual", "mix_day_night_scores"),
        "aggregate_sequence_scores": ("pet_behavior_clip.contextual", "aggregate_sequence_scores"),
        "compute_ece_from_labeled_scores": ("pet_behavior_clip.contextual", "compute_ece_from_labeled_scores"),
    }
    if name in _lazy:
        import importlib

        module_name, attr = _lazy[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module 'pet_behavior_clip' has no attribute {name!r}")
