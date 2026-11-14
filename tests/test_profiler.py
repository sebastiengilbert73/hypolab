"""Unit tests for the Polars dataset profiler and quality alert detector."""

from __future__ import annotations

import numpy as np
import polars as pl

from hypolab.models import DistributionType
from hypolab.profiler.engine import profile_dataset


def test_profile_dataset_types_and_distributions() -> None:
    rng = np.random.default_rng(123)
    n = 300
    normal_col = rng.normal(100.0, 10.0, size=n).tolist()
    skewed_col = rng.exponential(scale=2.0, size=n).tolist()
    skewed_col[0] = None  # inject missing value
    cat_col = rng.choice(["A", "B", "C"], size=n).tolist()
    bin_col = rng.choice([True, False], size=n).tolist()
    const_col = [42] * n

    df = pl.DataFrame(
        {
            "normal_metric": normal_col,
            "skewed_metric": skewed_col,
            "segment": cat_col,
            "is_active": bin_col,
            "constant_val": const_col,
        }
    )

    profile = profile_dataset(df, dataset_name="synthetic_test.csv")
    assert profile.dataset_name == "synthetic_test.csv"
    assert profile.row_count == n
    assert profile.column_count == 5

    col_map = {c.name: c for c in profile.columns}
    assert col_map["normal_metric"].semantic_type == "numeric"
    assert col_map["normal_metric"].distribution_type == DistributionType.NORMAL

    assert col_map["skewed_metric"].semantic_type == "numeric"
    assert col_map["skewed_metric"].missing_count == 1
    assert col_map["skewed_metric"].distribution_type == DistributionType.SKEWED

    assert col_map["segment"].semantic_type == "categorical"
    assert col_map["segment"].unique_count == 3

    assert col_map["is_active"].semantic_type == "boolean"
    assert col_map["constant_val"].distribution_type == DistributionType.CONSTANT

    alert_types = {a.issue_type for a in profile.quality_alerts}
    assert "constant_column" in alert_types

    llm_summary = profile.to_llm_summary()
    assert "synthetic_test.csv" in llm_summary
    assert "normal_metric" in llm_summary
