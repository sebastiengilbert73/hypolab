"""Data quality checks: outliers, missingness, constant columns, and collinearity."""

from __future__ import annotations

import polars as pl
from hypolab.models import ColumnProfile, CorrelationPair, QualityAlert


def count_iqr_outliers(series: pl.Series, q25: float | None, q75: float | None) -> int:
    """Count outliers outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    if q25 is None or q75 is None:
        return 0
    iqr = q75 - q25
    if iqr <= 0:
        return 0
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return 0
    return int(((non_null < lower) | (non_null > upper)).sum())


def detect_quality_alerts(
    row_count: int,
    duplicate_row_count: int,
    columns: list[ColumnProfile],
    correlations: list[CorrelationPair],
) -> list[QualityAlert]:
    """Generate actionable quality alerts from column profiles and correlations."""
    alerts: list[QualityAlert] = []

    if row_count < 30:
        alerts.append(
            QualityAlert(
                severity="WARNING",
                issue_type="small_sample_size",
                description=(
                    f"Dataset has only {row_count} rows (< 30). Non-parametric tests "
                    "and exact tests should be preferred due to low statistical power."
                ),
            )
        )

    if duplicate_row_count > 0:
        dup_pct = (duplicate_row_count / max(row_count, 1)) * 100.0
        severity = "WARNING" if dup_pct > 5.0 else "INFO"
        alerts.append(
            QualityAlert(
                severity=severity,
                issue_type="duplicate_rows",
                description=f"{duplicate_row_count} duplicate rows detected ({dup_pct:.1f}% of dataset).",
            )
        )

    for col in columns:
        if col.missing_pct >= 50.0:
            alerts.append(
                QualityAlert(
                    severity="CRITICAL",
                    column=col.name,
                    issue_type="high_missingness",
                    description=f"Column '{col.name}' has {col.missing_pct:.1f}% missing values.",
                )
            )
        elif col.missing_pct >= 10.0:
            alerts.append(
                QualityAlert(
                    severity="WARNING",
                    column=col.name,
                    issue_type="moderate_missingness",
                    description=f"Column '{col.name}' has {col.missing_pct:.1f}% missing values.",
                )
            )

        if col.unique_count <= 1 and row_count > 1:
            alerts.append(
                QualityAlert(
                    severity="WARNING",
                    column=col.name,
                    issue_type="constant_column",
                    description=f"Column '{col.name}' is constant (unique_count={col.unique_count}) and carries no variance.",
                )
            )

        if col.outlier_count_iqr > 0 and col.count > 0:
            outlier_pct = (col.outlier_count_iqr / col.count) * 100.0
            if outlier_pct >= 5.0:
                alerts.append(
                    QualityAlert(
                        severity="INFO",
                        column=col.name,
                        issue_type="heavy_tails_or_outliers",
                        description=(
                            f"Column '{col.name}' has {col.outlier_count_iqr} IQR outliers "
                            f"({outlier_pct:.1f}% of valid observations)."
                        ),
                    )
                )

    for corr in correlations:
        if corr.abs_max >= 0.95:
            alerts.append(
                QualityAlert(
                    severity="WARNING",
                    column=f"{corr.col_a}, {corr.col_b}",
                    issue_type="multicollinearity",
                    description=(
                        f"Near-collinear relationship between '{corr.col_a}' and '{corr.col_b}' "
                        f"(r={corr.pearson_r:.3f}, rho={corr.spearman_rho:.3f}). Avoid including both as predictors."
                    ),
                )
            )

    return alerts
