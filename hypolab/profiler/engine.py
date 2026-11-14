"""Fast dataset loading and statistical profiling engine powered by Polars and SciPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from hypolab.models import (
    ColumnProfile,
    CorrelationPair,
    DatasetProfile,
    DistributionType,
)
from hypolab.profiler.quality import count_iqr_outliers, detect_quality_alerts


def load_dataset(source: str | Path | pl.DataFrame | pd.DataFrame) -> tuple[pl.DataFrame, str]:
    """Load a dataset from a file path, Polars DataFrame, or Pandas DataFrame."""
    if isinstance(source, pl.DataFrame):
        return source, "in_memory_dataframe"
    if isinstance(source, pd.DataFrame):
        return pl.from_pandas(source), "in_memory_dataframe"

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(path, try_parse_dates=True, ignore_errors=True)
    elif suffix in (".parquet", ".pq"):
        df = pl.read_parquet(path)
    elif suffix in (".json", ".ndjson", ".jsonl"):
        if suffix == ".json":
            df = pl.read_json(path)
        else:
            df = pl.read_ndjson(path)
    else:
        df = pl.read_csv(path, try_parse_dates=True, ignore_errors=True)

    return df, path.name


def _infer_numeric_distribution(series: pl.Series, unique_count: int) -> tuple[DistributionType, float | None, float | None]:
    """Infer distribution shape, skewness, and kurtosis for a numeric series."""
    clean = series.drop_nulls().cast(pl.Float64)
    n = len(clean)
    if n == 0 or unique_count <= 1:
        return DistributionType.CONSTANT, 0.0, 0.0
    if unique_count == 2:
        return DistributionType.BINARY, 0.0, 0.0

    arr = clean.to_numpy()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return DistributionType.NORMAL, 0.0, 0.0

    skew_val = float(stats.skew(arr, bias=False)) if len(arr) > 3 else 0.0
    kurt_val = float(stats.kurtosis(arr, bias=False)) if len(arr) > 3 else 0.0

    if not np.isfinite(skew_val):
        skew_val = 0.0
    if not np.isfinite(kurt_val):
        kurt_val = 0.0

    if abs(skew_val) > 1.0:
        return DistributionType.SKEWED, skew_val, kurt_val

    # Check for uniform vs multimodal vs normal
    if kurt_val < -1.1:
        # Strong negative excess kurtosis indicates either uniform or bimodal
        hist, _ = np.histogram(arr, bins=min(15, max(5, int(np.sqrt(len(arr))))))
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks += 1
        if peaks >= 2:
            return DistributionType.BIMODAL_OR_MULTIMODAL, skew_val, kurt_val
        return DistributionType.UNIFORM, skew_val, kurt_val

    return DistributionType.NORMAL, skew_val, kurt_val


def _profile_column(series: pl.Series, row_count: int) -> ColumnProfile:
    """Compute a detailed ColumnProfile for a single Polars Series."""
    dtype_str = str(series.dtype)
    missing_count = int(series.null_count())
    valid_count = row_count - missing_count
    missing_pct = (missing_count / max(row_count, 1)) * 100.0
    non_null = series.drop_nulls()
    unique_count = int(non_null.n_unique())

    sample_vals: list[Any] = non_null.head(5).to_list()

    # Check if identifier column
    col_lower = series.name.lower()
    is_id_name = col_lower == "id" or col_lower.endswith("_id") or col_lower.startswith("id_")
    if unique_count <= 1 and valid_count > 0:
        return ColumnProfile(
            name=series.name,
            dtype=dtype_str,
            semantic_type="constant",
            count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            distribution_type=DistributionType.CONSTANT,
            sample_values=sample_vals,
        )

    if series.dtype.is_numeric() and not isinstance(series.dtype, pl.Boolean):
        if is_id_name and unique_count == valid_count and valid_count > 20:
            return ColumnProfile(
                name=series.name,
                dtype=dtype_str,
                semantic_type="identifier",
                count=valid_count,
                missing_count=missing_count,
                missing_pct=round(missing_pct, 2),
                unique_count=unique_count,
                distribution_type=DistributionType.IDENTIFIER,
                sample_values=sample_vals,
            )

        dist_type, skew_val, kurt_val = _infer_numeric_distribution(series, unique_count)
        mean_val = float(non_null.mean()) if valid_count > 0 else None
        std_val = float(non_null.std()) if valid_count > 1 else 0.0
        min_val = float(non_null.min()) if valid_count > 0 else None
        q25_val = float(non_null.quantile(0.25)) if valid_count > 0 else None
        med_val = float(non_null.median()) if valid_count > 0 else None
        q75_val = float(non_null.quantile(0.75)) if valid_count > 0 else None
        max_val = float(non_null.max()) if valid_count > 0 else None
        outliers = count_iqr_outliers(non_null, q25_val, q75_val)

        return ColumnProfile(
            name=series.name,
            dtype=dtype_str,
            semantic_type="numeric",
            count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            distribution_type=dist_type,
            mean=mean_val,
            std=std_val,
            min=min_val,
            q25=q25_val,
            median=med_val,
            q75=q75_val,
            max=max_val,
            skewness=skew_val,
            kurtosis=kurt_val,
            outlier_count_iqr=outliers,
            sample_values=sample_vals,
        )

    if isinstance(series.dtype, pl.Boolean):
        vc = non_null.cast(pl.String).value_counts(sort=True)
        top_cats = {
            str(row[0]): int(row[1])
            for row in vc.head(10).iter_rows()
        }
        return ColumnProfile(
            name=series.name,
            dtype=dtype_str,
            semantic_type="boolean",
            count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            distribution_type=DistributionType.BINARY,
            top_categories=top_cats,
            sample_values=sample_vals,
        )

    if series.dtype.is_temporal():
        min_dt = str(non_null.min()) if valid_count > 0 else None
        max_dt = str(non_null.max()) if valid_count > 0 else None
        return ColumnProfile(
            name=series.name,
            dtype=dtype_str,
            semantic_type="datetime",
            count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            distribution_type=DistributionType.TEMPORAL,
            min=min_dt,
            max=max_dt,
            sample_values=[str(v) for v in sample_vals],
        )

    # String / Categorical
    if (is_id_name or unique_count == valid_count) and valid_count > 30:
        return ColumnProfile(
            name=series.name,
            dtype=dtype_str,
            semantic_type="identifier",
            count=valid_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            distribution_type=DistributionType.IDENTIFIER,
            sample_values=sample_vals,
        )

    vc = non_null.cast(pl.String).value_counts(sort=True)
    top_cats = {
        str(row[0]): int(row[1])
        for row in vc.head(10).iter_rows()
    }
    dist = DistributionType.BINARY if unique_count == 2 else DistributionType.CATEGORICAL
    return ColumnProfile(
        name=series.name,
        dtype=dtype_str,
        semantic_type="categorical",
        count=valid_count,
        missing_count=missing_count,
        missing_pct=round(missing_pct, 2),
        unique_count=unique_count,
        distribution_type=dist,
        top_categories=top_cats,
        sample_values=sample_vals,
    )


def _compute_top_correlations(df: pl.DataFrame, columns: list[ColumnProfile], max_pairs: int = 15) -> list[CorrelationPair]:
    """Compute pairwise Pearson and Spearman correlations for numeric columns."""
    num_cols = [
        c.name
        for c in columns
        if c.semantic_type == "numeric" and c.unique_count > 1
    ]
    if len(num_cols) < 2:
        return []

    pairs: list[CorrelationPair] = []
    sub_pd = df.select(num_cols).to_pandas()
    pearson_mat = sub_pd.corr(method="pearson")
    spearman_mat = sub_pd.corr(method="spearman")

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            ca, cb = num_cols[i], num_cols[j]
            r = float(pearson_mat.loc[ca, cb])
            rho = float(spearman_mat.loc[ca, cb])
            if np.isnan(r):
                r = 0.0
            if np.isnan(rho):
                rho = 0.0
            abs_max = max(abs(r), abs(rho))
            pairs.append(
                CorrelationPair(
                    col_a=ca,
                    col_b=cb,
                    pearson_r=round(r, 4),
                    spearman_rho=round(rho, 4),
                    abs_max=round(abs_max, 4),
                )
            )

    pairs.sort(key=lambda p: p.abs_max, reverse=True)
    return pairs[:max_pairs]


def profile_dataset(
    source: str | Path | pl.DataFrame | pd.DataFrame,
    dataset_name: str | None = None,
) -> DatasetProfile:
    """Profile any tabular dataset into a structured DatasetProfile."""
    df, inferred_name = load_dataset(source)
    name = dataset_name or inferred_name
    row_count = df.height
    col_count = df.width
    dup_count = int(df.is_duplicated().sum()) if row_count > 0 else 0
    mem_bytes = int(df.estimated_size())

    col_profiles = [_profile_column(df[col], row_count) for col in df.columns]
    top_corrs = _compute_top_correlations(df, col_profiles)
    alerts = detect_quality_alerts(row_count, dup_count, col_profiles, top_corrs)

    return DatasetProfile(
        dataset_name=name,
        row_count=row_count,
        column_count=col_count,
        duplicate_row_count=dup_count,
        memory_bytes=mem_bytes,
        columns=col_profiles,
        top_correlations=top_corrs,
        quality_alerts=alerts,
    )
