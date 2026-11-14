"""Data profiling and quality analysis package."""

from hypolab.profiler.engine import profile_dataset, load_dataset
from hypolab.profiler.quality import detect_quality_alerts, count_iqr_outliers

__all__ = [
    "profile_dataset",
    "load_dataset",
    "detect_quality_alerts",
    "count_iqr_outliers",
]
