"""Unit tests for deterministic statistical testing tools."""

from __future__ import annotations

import numpy as np
import polars as pl

from hypolab.benchmarks import (
    generate_saas_churn_dataset,
    generate_simpsons_paradox_dataset,
)
from hypolab.models import ConclusionStatus
from hypolab.tools.statistical_tests import (
    compare_multiple_groups,
    compare_two_groups,
    multivariate_regression,
    test_categorical_association,
    test_correlation,
)


def test_compare_two_groups_detects_real_difference_and_rejects_noise() -> None:
    df = generate_saas_churn_dataset(n=400, seed=42)

    # True difference: support_tickets_30d by churned
    res_true = compare_two_groups(df, group_col="churned", target_col="support_tickets_30d")
    assert res_true.conclusion == ConclusionStatus.VALIDATED
    assert res_true.p_value is not None and res_true.p_value < 1e-5
    assert res_true.effect_size is not None and res_true.effect_size > 0.3
    assert res_true.figure_json is not None

    # Noise: lucky_number_noise by churned
    res_noise = compare_two_groups(df, group_col="churned", target_col="lucky_number_noise")
    assert res_noise.conclusion == ConclusionStatus.REFUTED


def test_compare_multiple_groups_anova_or_kruskal() -> None:
    df = generate_saas_churn_dataset(n=400, seed=42)
    res = compare_multiple_groups(df, group_col="plan_tier", target_col="monthly_spend_usd")
    assert res.conclusion == ConclusionStatus.VALIDATED
    assert res.p_value is not None and res.p_value < 1e-10
    assert res.effect_size is not None and res.effect_size > 0.5


def test_categorical_association_chi2_and_fisher() -> None:
    df = generate_saas_churn_dataset(n=400, seed=42)
    res = test_categorical_association(df, col_a="contract_type", col_b="churned")
    assert res.conclusion == ConclusionStatus.VALIDATED
    assert res.p_value is not None and res.p_value < 1e-4
    assert res.effect_size_name == "Cramér's V"

    # Small 2x2 table triggering Fisher's Exact Test
    small_df = pl.DataFrame(
        {
            "treatment": ["Drug"] * 6 + ["Placebo"] * 6,
            "recovered": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes"],
        }
    )
    res_fisher = test_categorical_association(small_df, col_a="treatment", col_b="recovered")
    assert "Fisher" in res_fisher.test_used


def test_correlation_detects_simpsons_paradox() -> None:
    df = generate_simpsons_paradox_dataset(n_per_group=120, seed=42)
    res = test_correlation(
        df,
        col_x="years_experience",
        col_y="salary_k",
        control_col="department",
    )
    assert res.additional_metrics["simpsons_paradox_detected"] is True
    assert "SIMPSON'S PARADOX DETECTED" in res.interpretation


def test_multivariate_regression_ols_and_logit() -> None:
    df = generate_saas_churn_dataset(n=400, seed=42)

    res_ols = multivariate_regression(
        df,
        target="monthly_spend_usd",
        features=["plan_tier", "usage_hours_weekly"],
        family="ols",
    )
    assert res_ols.conclusion == ConclusionStatus.VALIDATED
    assert res_ols.effect_size is not None and res_ols.effect_size > 0.5

    res_logit = multivariate_regression(
        df,
        target="churned",
        features=["contract_type", "usage_hours_weekly"],
        family="logit",
    )
    assert res_logit.conclusion == ConclusionStatus.VALIDATED
    assert "Logistic" in res_logit.test_used
