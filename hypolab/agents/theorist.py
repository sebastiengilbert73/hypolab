"""Theorist Agent: Formulates falsifiable statistical hypotheses (H0 vs H1) from a dataset profile."""

from __future__ import annotations

from hypolab.agents.llm_client import call_llm_json, is_llm_configured
from hypolab.config import HypoLabConfig
from hypolab.models import (
    DatasetProfile,
    Hypothesis,
    ProposedMethod,
)

THEORIST_SYSTEM_PROMPT = """\
You are an elite quantitative research scientist (Theorist Agent) in HypoLab.
Given a statistical profile of a dataset and a research goal, formulate falsifiable, high-value scientific hypotheses.

Rules:
1. Every hypothesis MUST specify:
   - id: "HYP-001", "HYP-002", etc.
   - title: Short descriptive title
   - statement: Clear alternative hypothesis (H1) predicting a specific relationship or difference
   - null_hypothesis: Formal null hypothesis (H0) of no effect / independence
   - rationale: Statistical reasoning grounded in the dataset profile
   - variables: List of primary column names involved (must match exact dataset column names)
   - confounders: List of 0-2 potential confounding columns in the dataset
   - proposed_method: One of:
     * "compare_two_groups" (tool_params: {"group_col": "<cat_col>", "target_col": "<num_col>"})
     * "compare_multiple_groups" (tool_params: {"group_col": "<cat_col>", "target_col": "<num_col>"})
     * "test_categorical_association" (tool_params: {"col_a": "<cat_col_1>", "col_b": "<cat_col_2>"})
     * "test_correlation" (tool_params: {"col_x": "<num_col_1>", "col_y": "<num_col_2>", "control_col": "<optional_confounder>"})
     * "multivariate_regression" (tool_params: {"target": "<target_col>", "features": ["<col1>", "<col2>"]})
     * "python_sandbox" (tool_params: {"code": "<valid python code assigning dict to result and plotly figure to fig>"})
   - priority: 1 (highest) to 5 (lowest)

2. Return ONLY a JSON object with a single key "hypotheses" containing the list of hypothesis objects.
"""


def generate_hypotheses(
    profile: DatasetProfile,
    research_goal: str = "Discover statistically significant relationships, group differences, and potential confounding effects.",
    config: HypoLabConfig | None = None,
    use_llm: bool = True,
) -> list[Hypothesis]:
    """Generate falsifiable statistical hypotheses using LLM or autonomous heuristic engine."""
    cfg = config or HypoLabConfig()

    if use_llm and is_llm_configured(cfg):
        try:
            user_prompt = (
                f"Research Goal: {research_goal}\n"
                f"Maximum Hypotheses to Generate: {cfg.max_hypotheses}\n\n"
                f"Dataset Profile:\n{profile.to_llm_summary()}\n\n"
                "Generate diverse, rigorous hypotheses covering correlations, group comparisons, "
                "categorical associations, and multivariate confounding checks."
            )
            raw = call_llm_json(THEORIST_SYSTEM_PROMPT, user_prompt, cfg)
            items = raw.get("hypotheses", raw) if isinstance(raw, dict) else raw
            valid_cols = {c.name for c in profile.columns}
            parsed: list[Hypothesis] = []
            for idx, item in enumerate(items[: cfg.max_hypotheses], start=1):
                if not isinstance(item, dict):
                    continue
                item.setdefault("id", f"HYP-{idx:03d}")
                hyp = Hypothesis.model_validate(item)
                if all(v in valid_cols for v in hyp.variables):
                    parsed.append(hyp)
            if parsed:
                return parsed
        except Exception:
            # Fall back cleanly to deterministic heuristic hypothesis generator
            pass

    return generate_heuristic_hypotheses(profile, research_goal, max_hypotheses=cfg.max_hypotheses)


def generate_heuristic_hypotheses(
    profile: DatasetProfile,
    research_goal: str = "",
    max_hypotheses: int = 6,
) -> list[Hypothesis]:
    """Deterministic statistical hypothesis generator based on dataset structure and correlations."""
    hypotheses: list[Hypothesis] = []
    num_cols = [c for c in profile.columns if c.semantic_type == "numeric" and c.unique_count > 2]
    bin_cols = [
        c
        for c in profile.columns
        if c.semantic_type in ("categorical", "boolean", "numeric") and c.unique_count == 2
    ]
    cat_cols = [
        c
        for c in profile.columns
        if c.semantic_type == "categorical" and 2 < c.unique_count <= 12
    ]
    all_cat_cols = bin_cols + cat_cols

    def _next_id() -> str:
        return f"HYP-{len(hypotheses) + 1:03d}"

    # 1. Top correlation hypotheses (with potential confounder stratification)
    for corr in profile.top_correlations[:2]:
        if len(hypotheses) >= max_hypotheses:
            break
        confounder_col = all_cat_cols[0].name if all_cat_cols else (
            next((c.name for c in num_cols if c.name not in (corr.col_a, corr.col_b)), None)
        )
        confounders = [confounder_col] if confounder_col else []
        direction = "positively" if (corr.pearson_r + corr.spearman_rho) >= 0 else "negatively"
        is_nonlinear = abs(corr.spearman_rho) - abs(corr.pearson_r) > 0.12
        rel_type = "monotonic non-linear" if is_nonlinear else "linear"

        hypotheses.append(
            Hypothesis(
                id=_next_id(),
                title=f"Relationship between {corr.col_a} and {corr.col_b}",
                statement=(
                    f"'{corr.col_a}' is {direction} associated with '{corr.col_b}' via a {rel_type} relationship"
                    + (f", even when accounting for '{confounder_col}'." if confounder_col else ".")
                ),
                null_hypothesis=f"There is no correlation (r = 0, ρ = 0) between '{corr.col_a}' and '{corr.col_b}'.",
                rationale=(
                    f"Initial profiling indicates Pearson r={corr.pearson_r:.3f} and Spearman ρ={corr.spearman_rho:.3f}."
                ),
                variables=[corr.col_a, corr.col_b],
                confounders=confounders,
                proposed_method=ProposedMethod.CORRELATION_TEST,
                tool_params={
                    "col_x": corr.col_a,
                    "col_y": corr.col_b,
                    "control_col": confounder_col,
                },
                priority=1,
            )
        )

    # 2. Binary two-group comparisons on numeric targets
    for bcol in bin_cols:
        for ncol in num_cols:
            if bcol.name == ncol.name or len(hypotheses) >= max_hypotheses:
                continue
            other_num = [c.name for c in num_cols if c.name != ncol.name][:1]
            hypotheses.append(
                Hypothesis(
                    id=_next_id(),
                    title=f"Difference in {ncol.name} by {bcol.name}",
                    statement=f"The distribution of '{ncol.name}' differs significantly between the two groups of '{bcol.name}'.",
                    null_hypothesis=f"The mean/median of '{ncol.name}' is identical across both groups of '{bcol.name}'.",
                    rationale=(
                        f"'{bcol.name}' is a binary grouping factor and '{ncol.name}' is a continuous outcome "
                        f"(mean={ncol.mean:.3g}, std={ncol.std:.3g})."
                    ),
                    variables=[bcol.name, ncol.name],
                    confounders=other_num,
                    proposed_method=ProposedMethod.COMPARE_TWO_GROUPS,
                    tool_params={"group_col": bcol.name, "target_col": ncol.name},
                    priority=2,
                )
            )
            break

    # 3. Multi-group comparison (ANOVA / Kruskal-Wallis)
    for ccol in cat_cols:
        for ncol in num_cols:
            if len(hypotheses) >= max_hypotheses:
                break
            hypotheses.append(
                Hypothesis(
                    id=_next_id(),
                    title=f"Variation of {ncol.name} across {ccol.name} categories",
                    statement=f"'{ncol.name}' varies significantly across the {ccol.unique_count} categories of '{ccol.name}'.",
                    null_hypothesis=f"All categories of '{ccol.name}' have the same central tendency for '{ncol.name}'.",
                    rationale=f"Testing whether categorical segment '{ccol.name}' explains variance in '{ncol.name}'.",
                    variables=[ccol.name, ncol.name],
                    confounders=[],
                    proposed_method=ProposedMethod.COMPARE_MULTIPLE_GROUPS,
                    tool_params={"group_col": ccol.name, "target_col": ncol.name},
                    priority=2,
                )
            )
            break

    # 4. Categorical contingency test (Chi-Square / Fisher)
    if len(all_cat_cols) >= 2 and len(hypotheses) < max_hypotheses:
        ca, cb = all_cat_cols[0], all_cat_cols[1]
        hypotheses.append(
            Hypothesis(
                id=_next_id(),
                title=f"Association between {ca.name} and {cb.name}",
                statement=f"Categorical variables '{ca.name}' and '{cb.name}' are statistically dependent.",
                null_hypothesis=f"'{ca.name}' and '{cb.name}' are independent in the population.",
                rationale=f"Evaluating contingency structure between '{ca.name}' ({ca.unique_count} levels) and '{cb.name}' ({cb.unique_count} levels).",
                variables=[ca.name, cb.name],
                confounders=[],
                proposed_method=ProposedMethod.CATEGORICAL_ASSOCIATION,
                tool_params={"col_a": ca.name, "col_b": cb.name},
                priority=3,
            )
        )

    # 5. Multivariate regression controlling for confounders
    if len(num_cols) >= 2 and len(hypotheses) < max_hypotheses:
        target_col = num_cols[-1].name
        predictor_cols = [c.name for c in num_cols[:-1][:3]]
        if all_cat_cols:
            predictor_cols.append(all_cat_cols[0].name)
        hypotheses.append(
            Hypothesis(
                id=_next_id(),
                title=f"Multivariate drivers of {target_col}",
                statement=(
                    f"Predictors {predictor_cols} jointly explain significant variance in '{target_col}' "
                    "after controlling for mutual confounding."
                ),
                null_hypothesis=f"All regression coefficients predicting '{target_col}' from {predictor_cols} equal zero.",
                rationale="Multivariate control is required to isolate direct effects from spurious confounding.",
                variables=[target_col] + predictor_cols,
                confounders=predictor_cols[1:],
                proposed_method=ProposedMethod.MULTIVARIATE_REGRESSION,
                tool_params={
                    "target": target_col,
                    "features": predictor_cols,
                    "family": "auto",
                },
                priority=2,
            )
        )

    # 6. Additional correlation or polynomial non-linear check in Python Sandbox
    if len(num_cols) >= 2 and len(hypotheses) < max_hypotheses:
        cx, cy = num_cols[0].name, num_cols[1].name
        sandbox_code = (
            f"x = df[{cx!r}].dropna().astype(float)\n"
            f"y = df.loc[x.index, {cy!r}].dropna().astype(float)\n"
            f"x = x.loc[y.index]\n"
            f"X_lin = sm.add_constant(x)\n"
            f"X_quad = sm.add_constant(np.column_stack([x, x**2]))\n"
            f"m_lin = sm.OLS(y, X_lin).fit()\n"
            f"m_quad = sm.OLS(y, X_quad).fit()\n"
            f"quad_p = float(m_quad.pvalues[2]) if len(m_quad.pvalues) > 2 else 1.0\n"
            f"r2_gain = float(m_quad.rsquared_adj - m_lin.rsquared_adj)\n"
            f"result = {{\n"
            f"    'test_used': 'Quadratic Polynomial OLS Regression (Sandbox)',\n"
            f"    'statistic_name': 't (quadratic term)',\n"
            f"    'statistic_value': float(m_quad.tvalues[2]) if len(m_quad.tvalues) > 2 else 0.0,\n"
            f"    'p_value': quad_p,\n"
            f"    'effect_size_name': 'Adjusted R²',\n"
            f"    'effect_size': float(max(m_quad.rsquared_adj, 0.0)),\n"
            f"    'r2_linear': float(m_lin.rsquared_adj),\n"
            f"    'r2_quadratic': float(m_quad.rsquared_adj),\n"
            f"    'r2_gain': r2_gain,\n"
            f"    'sample_size': int(len(x)),\n"
            f"    'interpretation': f'Quadratic model Adj-R²={{m_quad.rsquared_adj:.3f}} vs Linear Adj-R²={{m_lin.rsquared_adj:.3f}} (ΔR²={{r2_gain:+.3f}}, quad p={{quad_p:.3e}}).'\n"
            f"}}\n"
            f"fig = px.scatter(df, x={cx!r}, y={cy!r}, title=f'Non-Linear Polynomial Fit: {cy} vs {cx} (Adj R²={{m_quad.rsquared_adj:.3f}})', template='plotly_white')\n"
        )
        hypotheses.append(
            Hypothesis(
                id=_next_id(),
                title=f"Non-linear (quadratic) curvature between {cx} and {cy}",
                statement=f"The relationship between '{cx}' and '{cy}' exhibits significant quadratic curvature (U-shaped or inverted-U).",
                null_hypothesis=f"The quadratic term β₂ in {cy} = β₀ + β₁·{cx} + β₂·{cx}² is zero.",
                rationale="Standard Pearson correlation misses non-monotonic U-shaped relationships.",
                variables=[cx, cy],
                confounders=[],
                proposed_method=ProposedMethod.PYTHON_SANDBOX,
                tool_params={"code": sandbox_code},
                priority=3,
            )
        )

    return hypotheses[:max_hypotheses]
