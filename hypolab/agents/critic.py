"""Critic Agent: Applies Benjamini-Hochberg FDR multiple-testing correction,
enforces minimum effect size rigor, and runs confounding control checks."""

from __future__ import annotations

import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

from hypolab.config import HypoLabConfig
from hypolab.models import (
    ConclusionStatus,
    ExperimentResult,
    Hypothesis,
    ProposedMethod,
)
from hypolab.tools.statistical_tests import multivariate_regression


def review_and_adjust_results(
    df: pl.DataFrame,
    hypotheses: list[Hypothesis],
    results: list[ExperimentResult],
    config: HypoLabConfig | None = None,
) -> list[ExperimentResult]:
    """Review experimental results, adjust p-values via Benjamini-Hochberg FDR,
    verify practical significance (effect size), and check confounders."""
    cfg = config or HypoLabConfig()
    hyp_map = {h.id: h for h in hypotheses}

    # 1. Collect valid p-values for Benjamini-Hochberg FDR correction
    valid_indices: list[int] = []
    raw_pvals: list[float] = []
    for idx, res in enumerate(results):
        if res.p_value is not None and np.isfinite(res.p_value):
            valid_indices.append(idx)
            raw_pvals.append(float(np.clip(res.p_value, 0.0, 1.0)))

    if raw_pvals:
        _, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=cfg.alpha, method="fdr_bh")
        for i, res_idx in enumerate(valid_indices):
            results[res_idx].adjusted_p_value = float(pvals_corrected[i])

    # 2. Evaluate each result against FDR alpha, effect size threshold, and confounding
    for res in results:
        if res.error_message:
            res.conclusion = ConclusionStatus.INCONCLUSIVE
            res.critic_notes = f"Execution incomplete: {res.error_message}"
            continue

        if res.p_value is None:
            res.conclusion = ConclusionStatus.INCONCLUSIVE
            res.critic_notes = "No p-value produced by the experiment."
            continue

        adj_p = res.adjusted_p_value if res.adjusted_p_value is not None else res.p_value
        eff = abs(res.effect_size) if res.effect_size is not None else 0.0

        # Check Simpson's paradox flag from correlation tool
        if res.additional_metrics.get("simpsons_paradox_detected"):
            res.conclusion = ConclusionStatus.REFUTED
            res.critic_notes = (
                "CRITICAL: Simpson's Paradox sign reversal detected! The apparent aggregate correlation "
                "reverses direction when stratified by the confounding variable. Aggregate H1 is refuted as spurious."
            )
            continue

        if adj_p >= cfg.alpha:
            if res.p_value < cfg.alpha:
                res.conclusion = ConclusionStatus.REFUTED
                res.critic_notes = (
                    f"Rejected after Benjamini-Hochberg FDR correction: raw p={res.p_value:.3e} < {cfg.alpha}, "
                    f"but FDR-adjusted p={adj_p:.3e} >= {cfg.alpha} (likely false positive from multiple testing)."
                )
            else:
                res.conclusion = ConclusionStatus.REFUTED
                res.critic_notes = (
                    f"Null hypothesis H0 cannot be rejected (FDR-adjusted p={adj_p:.3e} >= α={cfg.alpha})."
                )
            continue

        # Statistical significance passed; now verify practical significance (effect size)
        min_eff = _get_threshold_for_metric(res.effect_size_name, cfg.min_effect_size)
        if eff < min_eff:
            res.conclusion = ConclusionStatus.REFUTED
            res.critic_notes = (
                f"Statistically significant (adj p={adj_p:.3e}), but effect size is negligible "
                f"({res.effect_size_name}={eff:.4f} < threshold {min_eff:.2f}). Classified as trivial noise."
            )
            continue

        # Optional confounding control check for bivariate hypotheses with confounders
        hyp = hyp_map.get(res.hypothesis_id)
        if (
            hyp
            and hyp.confounders
            and hyp.proposed_method != ProposedMethod.MULTIVARIATE_REGRESSION
            and len(hyp.variables) >= 2
        ):
            conf_note, survives = _verify_against_confounders(df, hyp, cfg.alpha)
            if not survives:
                res.conclusion = ConclusionStatus.REFUTED
                res.critic_notes = (
                    f"Bivariate test passed (adj p={adj_p:.3e}), but failed confounding control: {conf_note}"
                )
                continue
            res.conclusion = ConclusionStatus.VALIDATED
            res.critic_notes = (
                f"Validated (FDR-adjusted p={adj_p:.3e} < {cfg.alpha}, {res.effect_size_name}={eff:.3f}). "
                f"{conf_note}"
            )
        else:
            res.conclusion = ConclusionStatus.VALIDATED
            res.critic_notes = (
                f"Validated with statistical rigor: FDR-adjusted p={adj_p:.3e} < {cfg.alpha} "
                f"and meaningful effect size ({res.effect_size_name}={eff:.3f})."
            )

    return results


def _get_threshold_for_metric(effect_size_name: str, default_min: float) -> float:
    """Return minimum practical significance threshold tailored to the effect size metric."""
    name_lower = effect_size_name.lower()
    if "r²" in name_lower or "eta-squared" in name_lower or "epsilon-squared" in name_lower:
        return min(default_min, 0.02)
    if "cramér" in name_lower or "cramer" in name_lower:
        return min(default_min, 0.08)
    return default_min


def _verify_against_confounders(
    df: pl.DataFrame,
    hypothesis: Hypothesis,
    alpha: float,
) -> tuple[str, bool]:
    """Run a quick multivariate regression check controlling for listed confounders."""
    # Determine predictor and target from hypothesis.variables
    var_a, var_b = hypothesis.variables[0], hypothesis.variables[1]
    if var_a not in df.columns or var_b not in df.columns:
        return "Confounder check skipped (columns missing).", True

    # Prefer numeric variable as target if one is categorical
    if df[var_b].dtype.is_numeric() and not isinstance(df[var_b].dtype, pl.Boolean):
        predictor, target = var_a, var_b
    elif df[var_a].dtype.is_numeric() and not isinstance(df[var_a].dtype, pl.Boolean):
        predictor, target = var_b, var_a
    else:
        predictor, target = var_a, var_b

    features = [predictor] + [c for c in hypothesis.confounders if c in df.columns and c not in (predictor, target)]
    if len(features) < 2:
        return "No distinct confounder columns available.", True

    reg_res = multivariate_regression(
        df,
        target=target,
        features=features,
        hypothesis_id=f"{hypothesis.id}-CTRL",
        alpha=alpha,
    )
    if reg_res.error_message or reg_res.p_value is None:
        return "Multivariate confounder check inconclusive.", True

    if reg_res.p_value >= alpha:
        return (
            f"When controlling for confounders {features[1:]} in multivariate regression, "
            f"the direct effect of '{predictor}' on '{target}' becomes non-significant (p={reg_res.p_value:.3e}). "
            "The bivariate association is explained by confounding.",
            False,
        )

    return (
        f"Effect survives multivariate control for {features[1:]} (controlled p={reg_res.p_value:.3e}).",
        True,
    )
