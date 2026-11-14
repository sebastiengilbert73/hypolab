"""Deterministic statistical testing suite with automatic parametric/non-parametric selection,
effect size computation, and interactive Plotly chart generation."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from hypolab.models import ConclusionStatus, ExperimentResult
from hypolab.tools.registry import register_tool


def _to_pandas(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df.copy()


def _check_normality(arr: np.ndarray, alpha: float = 0.05) -> bool:
    """Return True if array passes Shapiro-Wilk normality check."""
    clean = arr[np.isfinite(arr)]
    if len(clean) < 8:
        return False
    sample = clean[:5000] if len(clean) > 5000 else clean
    if np.std(sample) == 0:
        return False
    try:
        _, p_val = stats.shapiro(sample)
        return bool(p_val >= alpha)
    except Exception:
        return False


@register_tool(
    name="compare_two_groups",
    description="Compare a numeric target variable between two groups (automatically chooses Student t-test, Welch t-test, or Mann-Whitney U, and computes Cohen's d or rank-biserial r).",
    parameters={
        "group_col": "Categorical/binary column defining the groups",
        "target_col": "Numeric column to compare",
        "group_a": "Optional specific label for first group",
        "group_b": "Optional specific label for second group",
    },
)
def compare_two_groups(
    df: pl.DataFrame | pd.DataFrame,
    group_col: str,
    target_col: str,
    hypothesis_id: str = "HYP-000",
    group_a: Any = None,
    group_b: Any = None,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Compare two independent groups on a numeric outcome."""
    pdf = _to_pandas(df)[[group_col, target_col]].dropna()
    pdf[target_col] = pd.to_numeric(pdf[target_col], errors="coerce")
    pdf = pdf.dropna()

    counts = pdf[group_col].value_counts()
    if len(counts) < 2:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="compare_two_groups (Insufficient Groups)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=f"Column '{group_col}' has fewer than 2 non-empty groups.",
        )

    if group_a is None or group_b is None:
        top_two = counts.index[:2].tolist()
        group_a, group_b = top_two[0], top_two[1]

    sub_df = pdf[pdf[group_col].isin([group_a, group_b])].copy()
    sub_df[group_col] = sub_df[group_col].astype(str)
    ga_str, gb_str = str(group_a), str(group_b)

    arr_a = sub_df.loc[sub_df[group_col] == ga_str, target_col].to_numpy(dtype=float)
    arr_b = sub_df.loc[sub_df[group_col] == gb_str, target_col].to_numpy(dtype=float)

    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="compare_two_groups (Insufficient Sample)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=f"Group sizes too small: {ga_str} (n={n_a}), {gb_str} (n={n_b}).",
        )

    normal_a = _check_normality(arr_a)
    normal_b = _check_normality(arr_b)
    _, levene_p = stats.levene(arr_a, arr_b)
    equal_var = bool(levene_p >= alpha)

    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    med_a, med_b = float(np.median(arr_a)), float(np.median(arr_b))
    var_a, var_b = float(np.var(arr_a, ddof=1)), float(np.var(arr_b, ddof=1))
    pooled_sd = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)))
    cohens_d = abs(mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

    # 95% CI on difference of means
    se_diff = float(np.sqrt(var_a / n_a + var_b / n_b))
    ci_low = (mean_a - mean_b) - 1.96 * se_diff
    ci_high = (mean_a - mean_b) + 1.96 * se_diff

    if normal_a and normal_b:
        if equal_var:
            stat_val, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=True)
            test_name = "Student's Independent t-test"
        else:
            stat_val, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=False)
            test_name = "Welch's t-test (unequal variances)"
        stat_label = "t"
        eff_name = "Cohen's d"
        eff_val = float(cohens_d)
    else:
        u_stat, p_val = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        stat_val = float(u_stat)
        test_name = "Mann-Whitney U test (non-parametric)"
        stat_label = "U"
        # Rank-biserial correlation r_rb = 2*U / (n_a * n_b) - 1
        r_rb = abs(2.0 * float(u_stat) / (n_a * n_b) - 1.0)
        eff_name = "Rank-Biserial r (with Cohen's d)"
        eff_val = float(max(r_rb, cohens_d))

    fig = px.box(
        sub_df,
        x=group_col,
        y=target_col,
        color=group_col,
        points="outliers",
        title=f"{target_col} by {group_col} ({test_name}: p={p_val:.2e}, {eff_name}={eff_val:.3f})",
        template="plotly_white",
    )

    code_str = (
        f"# Two-group comparison: {target_col} across {group_col} ({ga_str} vs {gb_str})\n"
        f"group_a = df[df['{group_col}'] == {ga_str!r}]['{target_col}'].dropna()\n"
        f"group_b = df[df['{group_col}'] == {gb_str!r}]['{target_col}'].dropna()\n"
        f"stat, p_val = stats.{'ttest_ind' if normal_a and normal_b else 'mannwhitneyu'}(group_a, group_b)"
    )

    conclusion = (
        ConclusionStatus.VALIDATED
        if (p_val < alpha and eff_val >= 0.1)
        else ConclusionStatus.REFUTED
    )

    interp = (
        f"Group '{ga_str}' (n={n_a}, mean={mean_a:.3g}, median={med_a:.3g}) vs "
        f"Group '{gb_str}' (n={n_b}, mean={mean_b:.3g}, median={med_b:.3g}). "
        f"{test_name} yielded {stat_label}={stat_val:.3f}, p={p_val:.3e}, {eff_name}={eff_val:.3f}."
    )

    return ExperimentResult(
        hypothesis_id=hypothesis_id,
        test_used=test_name,
        statistic_name=stat_label,
        statistic_value=round(float(stat_val), 4),
        p_value=float(p_val),
        effect_size_name=eff_name,
        effect_size=round(float(eff_val), 4),
        confidence_interval=(round(ci_low, 4), round(ci_high, 4)),
        sample_size=int(n_a + n_b),
        conclusion=conclusion,
        interpretation=interp,
        code_executed=code_str,
        additional_metrics={
            "group_a": ga_str,
            "group_b": gb_str,
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "median_a": round(med_a, 4),
            "median_b": round(med_b, 4),
            "cohens_d": round(cohens_d, 4),
            "levene_p": round(float(levene_p), 4),
        },
        figure_json=fig.to_json(),
    )


@register_tool(
    name="compare_multiple_groups",
    description="Compare a numeric target across 3 or more categorical groups using One-Way ANOVA or Kruskal-Wallis test with Eta-squared effect size and Tukey HSD post-hoc.",
    parameters={
        "group_col": "Categorical column defining the groups",
        "target_col": "Numeric target column",
    },
)
def compare_multiple_groups(
    df: pl.DataFrame | pd.DataFrame,
    group_col: str,
    target_col: str,
    hypothesis_id: str = "HYP-000",
    alpha: float = 0.05,
) -> ExperimentResult:
    """Compare 3+ independent groups on a numeric outcome."""
    pdf = _to_pandas(df)[[group_col, target_col]].dropna()
    pdf[target_col] = pd.to_numeric(pdf[target_col], errors="coerce")
    pdf = pdf.dropna()

    # Keep top 10 groups with at least 3 observations
    counts = pdf[group_col].value_counts()
    valid_groups = counts[counts >= 3].index[:10].tolist()
    if len(valid_groups) < 2:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="compare_multiple_groups (Insufficient Groups)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=f"Column '{group_col}' needs at least 2 groups with >= 3 observations.",
        )

    sub_df = pdf[pdf[group_col].isin(valid_groups)].copy()
    sub_df[group_col] = sub_df[group_col].astype(str)
    arrays = [
         grp[target_col].to_numpy(dtype=float)
        for _, grp in sub_df.groupby(group_col)
    ]

    all_normal = all(_check_normality(a) for a in arrays)
    _, levene_p = stats.levene(*arrays)
    n_total = len(sub_df)
    k = len(arrays)

    # Eta-squared: SS_between / SS_total
    grand_mean = float(sub_df[target_col].mean())
    ss_total = float(np.sum((sub_df[target_col].to_numpy(dtype=float) - grand_mean) ** 2))
    ss_between = float(sum(len(a) * ((float(np.mean(a)) - grand_mean) ** 2) for a in arrays))
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    if all_normal and levene_p >= alpha:
        stat_val, p_val = stats.f_oneway(*arrays)
        test_name = "One-Way ANOVA"
        stat_label = "F"
        eff_name = "Eta-squared (η²)"
        eff_val = eta_sq
    else:
        stat_val, p_val = stats.kruskal(*arrays)
        test_name = "Kruskal-Wallis H-test"
        stat_label = "H"
        # Epsilon-squared for Kruskal-Wallis: H / ((n^2 - 1)/(n + 1)) = H / (n - 1)
        eps_sq = float(stat_val) / max(n_total - 1, 1)
        eff_name = "Epsilon-squared (ε²)"
        eff_val = max(eps_sq, eta_sq)

    # Post-hoc Tukey HSD summary
    tukey_pairs: list[str] = []
    try:
        tukey = pairwise_tukeyhsd(endog=sub_df[target_col], groups=sub_df[group_col], alpha=alpha)
        for row in tukey._results_table.data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = row
            if reject:
                tukey_pairs.append(f"{g1} vs {g2} (diff={meandiff:.3g}, p_adj={p_adj:.3g})")
    except Exception:
        pass

    fig = px.box(
        sub_df,
        x=group_col,
        y=target_col,
        color=group_col,
        title=f"{target_col} across {group_col} ({test_name}: p={p_val:.2e}, {eff_name}={eff_val:.3f})",
        template="plotly_white",
    )

    code_str = (
        f"# Multi-group comparison: {target_col} across {group_col} ({k} groups)\n"
        f"groups = [g['{target_col}'].values for _, g in df.groupby('{group_col}')]\n"
        f"stat, p_val = stats.{'f_oneway' if all_normal and levene_p >= alpha else 'kruskal'}(*groups)"
    )

    conclusion = (
        ConclusionStatus.VALIDATED
        if (p_val < alpha and eff_val >= 0.02)
        else ConclusionStatus.REFUTED
    )

    posthoc_text = (
        f" Significant pairwise differences (Tukey HSD): {'; '.join(tukey_pairs[:4])}."
        if tukey_pairs
        else ""
    )
    interp = (
        f"Compared {target_col} across {k} groups of '{group_col}' (N={n_total}). "
        f"{test_name}: {stat_label}={stat_val:.3f}, p={p_val:.3e}, {eff_name}={eff_val:.3f}.{posthoc_text}"
    )

    return ExperimentResult(
        hypothesis_id=hypothesis_id,
        test_used=test_name,
        statistic_name=stat_label,
        statistic_value=round(float(stat_val), 4),
        p_value=float(p_val),
        effect_size_name=eff_name,
        effect_size=round(float(eff_val), 4),
        sample_size=int(n_total),
        conclusion=conclusion,
        interpretation=interp,
        code_executed=code_str,
        additional_metrics={
            "num_groups": k,
            "eta_squared": round(eta_sq, 4),
            "significant_pairs": tukey_pairs[:5],
        },
        figure_json=fig.to_json(),
    )


@register_tool(
    name="test_categorical_association",
    description="Test independence between two categorical variables using Chi-Square or Fisher's Exact test and compute Cramér's V effect size.",
    parameters={
        "col_a": "First categorical column",
        "col_b": "Second categorical column",
    },
)
def test_categorical_association(
    df: pl.DataFrame | pd.DataFrame,
    col_a: str,
    col_b: str,
    hypothesis_id: str = "HYP-000",
    alpha: float = 0.05,
) -> ExperimentResult:
    """Test association between two categorical variables."""
    pdf = _to_pandas(df)[[col_a, col_b]].dropna().astype(str)
    if len(pdf) < 5:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="test_categorical_association (Insufficient Data)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message="Fewer than 5 valid rows.",
        )

    ct = pd.crosstab(pdf[col_a], pdf[col_b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="test_categorical_association (Degenerate Table)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=f"Contingency table shape is {ct.shape}; need at least 2x2.",
        )

    chi2, p_val, dof, expected = stats.chi2_contingency(ct)
    n = int(ct.to_numpy().sum())
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / max(n * min_dim, 1)))

    if ct.shape == (2, 2) and np.min(expected) < 5:
        odds_ratio, p_val = stats.fisher_exact(ct)
        test_name = "Fisher's Exact Test (2x2)"
        stat_label = "OR"
        stat_val = float(odds_ratio)
    else:
        test_name = "Pearson's Chi-Square Test of Independence"
        stat_label = "χ²"
        stat_val = float(chi2)

    # Stacked / grouped proportion chart
    prop_df = (
        ct.div(ct.sum(axis=1), axis=0)
        .reset_index()
        .melt(id_vars=col_a, var_name=col_b, value_name="proportion")
    )
    fig = px.bar(
        prop_df,
        x=col_a,
        y="proportion",
        color=col_b,
        barmode="group",
        title=f"Association: {col_a} vs {col_b} (p={p_val:.2e}, Cramér's V={cramers_v:.3f})",
        template="plotly_white",
    )

    code_str = (
        f"ct = pd.crosstab(df['{col_a}'], df['{col_b}'])\n"
        f"chi2, p_val, dof, ex = stats.chi2_contingency(ct)\n"
        f"cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))"
    )

    conclusion = (
        ConclusionStatus.VALIDATED
        if (p_val < alpha and cramers_v >= 0.08)
        else ConclusionStatus.REFUTED
    )

    interp = (
        f"Contingency analysis ({ct.shape[0]}x{ct.shape[1]}, N={n}) between '{col_a}' and '{col_b}'. "
        f"{test_name}: {stat_label}={stat_val:.3f}, p={p_val:.3e}, Cramér's V={cramers_v:.3f}."
    )

    return ExperimentResult(
        hypothesis_id=hypothesis_id,
        test_used=test_name,
        statistic_name=stat_label,
        statistic_value=round(stat_val, 4),
        p_value=float(p_val),
        effect_size_name="Cramér's V",
        effect_size=round(cramers_v, 4),
        sample_size=n,
        conclusion=conclusion,
        interpretation=interp,
        code_executed=code_str,
        additional_metrics={"dof": int(dof), "table_shape": list(ct.shape)},
        figure_json=fig.to_json(),
    )


@register_tool(
    name="test_correlation",
    description="Test linear (Pearson r) and monotonic (Spearman rho) correlation between two numeric variables, optionally stratifying/controlling for a third variable (to detect Simpson's Paradox or confounding).",
    parameters={
        "col_x": "First numeric column",
        "col_y": "Second numeric column",
        "control_col": "Optional confounding variable (categorical or numeric) to control/stratify for",
    },
)
def test_correlation(
    df: pl.DataFrame | pd.DataFrame,
    col_x: str,
    col_y: str,
    control_col: str | None = None,
    hypothesis_id: str = "HYP-000",
    alpha: float = 0.05,
) -> ExperimentResult:
    """Evaluate correlation between two numeric variables, optionally controlling for a third."""
    cols = [col_x, col_y] + ([control_col] if control_col and control_col in df.columns else [])
    pdf = _to_pandas(df)[cols].dropna()
    pdf[col_x] = pd.to_numeric(pdf[col_x], errors="coerce")
    pdf[col_y] = pd.to_numeric(pdf[col_y], errors="coerce")
    pdf = pdf.dropna()

    n = len(pdf)
    if n < 5:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="test_correlation (Insufficient Data)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message="Fewer than 5 valid numeric pairs.",
        )

    x_arr = pdf[col_x].to_numpy(dtype=float)
    y_arr = pdf[col_y].to_numpy(dtype=float)
    r_val, p_pearson = stats.pearsonr(x_arr, y_arr)
    rho_val, p_spearman = stats.spearmanr(x_arr, y_arr)

    # Check if monotonic non-linear (Spearman substantially stronger than Pearson)
    if abs(rho_val) - abs(r_val) > 0.12:
        primary_stat = float(rho_val)
        primary_p = float(p_spearman)
        test_name = "Spearman Rank Correlation (Monotonic)"
        stat_label = "ρ"
    else:
        primary_stat = float(r_val)
        primary_p = float(p_pearson)
        test_name = "Pearson & Spearman Correlation"
        stat_label = "r"

    extra_info = ""
    subgroup_corrs: dict[str, float] = {}
    simpsons_reversal = False

    if control_col and control_col in pdf.columns:
        # Check subgroup correlations if categorical or low-cardinality
        if pdf[control_col].nunique() <= 10:
            for grp_val, grp_df in pdf.groupby(control_col):
                if len(grp_df) >= 5:
                    sub_r, _ = stats.pearsonr(grp_df[col_x], grp_df[col_y])
                    if np.isfinite(sub_r):
                        subgroup_corrs[str(grp_val)] = round(float(sub_r), 4)
            if subgroup_corrs:
                # Check for Simpson's paradox sign reversal!
                signs = [np.sign(v) for v in subgroup_corrs.values() if abs(v) > 0.1]
                if signs and abs(r_val) > 0.1 and all(s != np.sign(r_val) for s in signs):
                    simpsons_reversal = True
                    mean_sub_r = float(np.mean(list(subgroup_corrs.values())))
                    extra_info = (
                        f" [SIMPSON'S PARADOX DETECTED: Overall r={r_val:.3f}, "
                        f"but within '{control_col}' strata mean r={mean_sub_r:.3f} ({subgroup_corrs})!]"
                    )
                else:
                    extra_info = f" Stratified by '{control_col}': {subgroup_corrs}."

    plot_df = pdf.sample(min(len(pdf), 2000), random_state=42) if len(pdf) > 2000 else pdf
    fig = px.scatter(
        plot_df,
        x=col_x,
        y=col_y,
        color=control_col if (control_col and control_col in plot_df.columns) else None,
        trendline="ols",
        title=f"{col_y} vs {col_x} (Pearson r={r_val:.3f}, Spearman ρ={rho_val:.3f}, p={primary_p:.2e})",
        template="plotly_white",
    )

    code_str = (
        f"r, p_pearson = stats.pearsonr(df['{col_x}'], df['{col_y}'])\n"
        f"rho, p_spearman = stats.spearmanr(df['{col_x}'], df['{col_y}'])"
    )

    eff_val = max(abs(float(r_val)), abs(float(rho_val)))
    conclusion = (
        ConclusionStatus.VALIDATED
        if (primary_p < alpha and eff_val >= 0.1)
        else ConclusionStatus.REFUTED
    )

    interp = (
        f"Correlation between '{col_x}' and '{col_y}' (N={n}): "
        f"Pearson r={r_val:.3f} (p={p_pearson:.3e}), Spearman ρ={rho_val:.3f} (p={p_spearman:.3e})."
        f"{extra_info}"
    )

    return ExperimentResult(
        hypothesis_id=hypothesis_id,
        test_used=test_name,
        statistic_name=stat_label,
        statistic_value=round(primary_stat, 4),
        p_value=primary_p,
        effect_size_name="|r| / |ρ|",
        effect_size=round(eff_val, 4),
        sample_size=n,
        conclusion=conclusion,
        interpretation=interp,
        code_executed=code_str,
        additional_metrics={
            "pearson_r": round(float(r_val), 4),
            "pearson_p": float(p_pearson),
            "spearman_rho": round(float(rho_val), 4),
            "spearman_p": float(p_spearman),
            "subgroup_correlations": subgroup_corrs,
            "simpsons_paradox_detected": simpsons_reversal,
        },
        figure_json=fig.to_json(),
    )


@register_tool(
    name="multivariate_regression",
    description="Fit an OLS or Logistic multivariate regression model to test whether predictors remain statistically significant when controlling for confounding variables.",
    parameters={
        "target": "Dependent variable (target column)",
        "features": "List of predictor and confounder column names",
        "family": "'ols', 'logit', or 'auto'",
    },
)
def multivariate_regression(
    df: pl.DataFrame | pd.DataFrame,
    target: str,
    features: list[str],
    family: str = "auto",
    hypothesis_id: str = "HYP-000",
    alpha: float = 0.05,
) -> ExperimentResult:
    """Run OLS or Logistic regression controlling for multiple predictors/confounders."""
    valid_features = [f for f in features if f in df.columns and f != target]
    if not valid_features or target not in df.columns:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="multivariate_regression (Invalid Columns)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=f"Target '{target}' or features {features} not found in dataset.",
        )

    pdf = _to_pandas(df)[[target] + valid_features].dropna()
    if len(pdf) < len(valid_features) + 5:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="multivariate_regression (Insufficient Sample)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message="Not enough rows to fit multivariate regression.",
        )

    y_raw = pdf[target]
    unique_y = y_raw.nunique()

    is_binary = (
        family == "logit"
        or (family == "auto" and (unique_y == 2 or y_raw.dtype == bool or y_raw.dtype == object))
    )

    if is_binary and unique_y == 2:
        vals = sorted(y_raw.unique().tolist(), key=lambda x: str(x))
        y = (y_raw == vals[1]).astype(float)
        model_type = "logit"
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        model_type = "ols"

    X_df = pd.get_dummies(pdf[valid_features], drop_first=True, dtype=float)
    valid_mask = y.notna() & X_df.notna().all(axis=1)
    y = y[valid_mask]
    X_df = X_df[valid_mask]
    X_const = sm.add_constant(X_df, has_constant="add")

    try:
        if model_type == "logit":
            model = sm.Logit(y, X_const).fit(disp=False, maxiter=100)
            r2_val = float(model.prsquared)
            r2_name = "McFadden's Pseudo-R²"
            model_p = float(model.llr_pvalue)
            stat_val = float(model.llr)
            stat_label = "LLR χ²"
            test_name = "Multivariate Logistic Regression"
        else:
            model = sm.OLS(y, X_const).fit()
            r2_val = float(model.rsquared_adj)
            r2_name = "Adjusted R²"
            model_p = float(model.f_pvalue)
            stat_val = float(model.fvalue)
            stat_label = "F"
            test_name = "Multivariate OLS Regression"
    except Exception as exc:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="multivariate_regression (Fit Error)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=str(exc),
        )

    params = model.params.drop("const", errors="ignore")
    pvals = model.pvalues.drop("const", errors="ignore")
    conf_int = model.conf_int().drop("const", errors="ignore")

    coef_records = []
    for feat_name in params.index:
        coef_records.append(
            {
                "feature": feat_name,
                "coef": round(float(params[feat_name]), 4),
                "p_value": float(pvals[feat_name]),
                "ci_low": round(float(conf_int.loc[feat_name, 0]), 4),
                "ci_high": round(float(conf_int.loc[feat_name, 1]), 4),
            }
        )

    # Primary predictor is the first feature in `valid_features`
    primary_prefix = valid_features[0]
    matching_pvals = [
        r["p_value"]
        for r in coef_records
        if r["feature"] == primary_prefix or r["feature"].startswith(f"{primary_prefix}_")
    ]
    primary_p = min(matching_pvals) if matching_pvals else model_p

    # Forest plot of coefficients
    coef_df = pd.DataFrame(coef_records)
    fig = go.Figure()
    if not coef_df.empty:
        fig.add_trace(
            go.Scatter(
                x=coef_df["coef"],
                y=coef_df["feature"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=coef_df["ci_high"] - coef_df["coef"],
                    arrayminus=coef_df["coef"] - coef_df["ci_low"],
                ),
                marker=dict(size=10, color="#2563eb"),
                name="Coefficient (95% CI)",
            )
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f"{test_name}: {target} ~ {' + '.join(valid_features)} ({r2_name}={r2_val:.3f})",
            xaxis_title="Coefficient Estimate (with 95% CI)",
            yaxis_title="Predictor",
            template="plotly_white",
        )

    coef_summary = ", ".join(
        f"{r['feature']}: β={r['coef']:.3g} (p={r['p_value']:.2e})"
        for r in coef_records
    )

    code_str = (
        f"X = sm.add_constant(pd.get_dummies(df[{valid_features!r}], drop_first=True, dtype=float))\n"
        f"model = sm.{'Logit' if model_type == 'logit' else 'OLS'}(df['{target}'], X).fit()\n"
        f"print(model.summary())"
    )

    conclusion = (
        ConclusionStatus.VALIDATED
        if (primary_p < alpha and max(r2_val, 0.0) >= 0.02)
        else ConclusionStatus.REFUTED
    )

    interp = (
        f"{test_name} predicting '{target}' from {valid_features} (N={len(y)}): "
        f"{r2_name}={r2_val:.3f}, overall model p={model_p:.3e}. "
        f"Predictors: {coef_summary}."
    )

    return ExperimentResult(
        hypothesis_id=hypothesis_id,
        test_used=test_name,
        statistic_name=stat_label,
        statistic_value=round(stat_val, 4) if np.isfinite(stat_val) else 0.0,
        p_value=float(primary_p),
        effect_size_name=r2_name,
        effect_size=round(max(r2_val, 0.0), 4),
        sample_size=int(len(y)),
        conclusion=conclusion,
        interpretation=interp,
        code_executed=code_str,
        additional_metrics={
            "model_p_value": model_p,
            "r2": round(r2_val, 4),
            "coefficients": coef_records,
        },
        figure_json=fig.to_json(),
    )
