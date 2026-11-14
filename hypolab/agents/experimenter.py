"""Experimenter Agent: Executes statistical tests or isolated Python sandbox scripts with self-healing."""

from __future__ import annotations

import polars as pl

from hypolab.agents.llm_client import call_llm_json, is_llm_configured
from hypolab.config import HypoLabConfig
from hypolab.models import (
    ConclusionStatus,
    DatasetProfile,
    ExperimentResult,
    Hypothesis,
    ProposedMethod,
)
from hypolab.tools.sandbox import run_in_sandbox
from hypolab.tools.statistical_tests import (
    compare_multiple_groups,
    compare_two_groups,
    multivariate_regression,
    test_categorical_association,
    test_correlation,
)

SANDBOX_SELF_HEAL_PROMPT = """\
You are the Experimenter Agent in HypoLab.
Write valid, self-contained Python code to test the given scientific hypothesis in a restricted sandbox.

Pre-loaded variables in the sandbox environment:
- `df`: pandas.DataFrame containing the dataset
- `pl_df`: polars.DataFrame containing the dataset
- `np` (numpy), `pd` (pandas), `pl` (polars), `stats` (scipy.stats), `sm` (statsmodels.api), `px` (plotly.express), `go` (plotly.graph_objects)

Security Rules (STRICT):
- DO NOT import `os`, `sys`, `subprocess`, `pathlib`, `open`, `eval`, `exec`, or any file/network I/O.
- DO NOT access attributes starting with `__` or file writer/reader methods (`to_csv`, `read_csv`, etc.).
- Assign a dictionary named `result` with keys:
  `test_used` (str), `statistic_name` (str), `statistic_value` (float), `p_value` (float),
  `effect_size_name` (str), `effect_size` (float), `sample_size` (int), `interpretation` (str).
- Optionally assign a Plotly figure to `fig`.

Return ONLY a JSON object: {"code": "<python_code>"}
"""


def run_experiment(
    df: pl.DataFrame,
    hypothesis: Hypothesis,
    profile: DatasetProfile | None = None,
    config: HypoLabConfig | None = None,
    use_llm: bool = True,
) -> ExperimentResult:
    """Execute the appropriate structured statistical tool or sandbox script for a hypothesis."""
    cfg = config or HypoLabConfig()
    method = hypothesis.proposed_method
    params = hypothesis.tool_params or {}
    vars_list = hypothesis.variables

    try:
        if method == ProposedMethod.COMPARE_TWO_GROUPS:
            group_col = params.get("group_col") or (vars_list[0] if len(vars_list) >= 2 else "")
            target_col = params.get("target_col") or (vars_list[1] if len(vars_list) >= 2 else "")
            return compare_two_groups(
                df,
                group_col=group_col,
                target_col=target_col,
                hypothesis_id=hypothesis.id,
                group_a=params.get("group_a"),
                group_b=params.get("group_b"),
                alpha=cfg.alpha,
            )

        if method == ProposedMethod.COMPARE_MULTIPLE_GROUPS:
            group_col = params.get("group_col") or (vars_list[0] if len(vars_list) >= 2 else "")
            target_col = params.get("target_col") or (vars_list[1] if len(vars_list) >= 2 else "")
            return compare_multiple_groups(
                df,
                group_col=group_col,
                target_col=target_col,
                hypothesis_id=hypothesis.id,
                alpha=cfg.alpha,
            )

        if method == ProposedMethod.CATEGORICAL_ASSOCIATION:
            col_a = params.get("col_a") or (vars_list[0] if len(vars_list) >= 2 else "")
            col_b = params.get("col_b") or (vars_list[1] if len(vars_list) >= 2 else "")
            return test_categorical_association(
                df,
                col_a=col_a,
                col_b=col_b,
                hypothesis_id=hypothesis.id,
                alpha=cfg.alpha,
            )

        if method == ProposedMethod.CORRELATION_TEST:
            col_x = params.get("col_x") or (vars_list[0] if len(vars_list) >= 2 else "")
            col_y = params.get("col_y") or (vars_list[1] if len(vars_list) >= 2 else "")
            control_col = params.get("control_col") or (
                hypothesis.confounders[0] if hypothesis.confounders else None
            )
            return test_correlation(
                df,
                col_x=col_x,
                col_y=col_y,
                control_col=control_col,
                hypothesis_id=hypothesis.id,
                alpha=cfg.alpha,
            )

        if method == ProposedMethod.MULTIVARIATE_REGRESSION:
            target = params.get("target") or (vars_list[0] if vars_list else "")
            features = params.get("features") or (vars_list[1:] + hypothesis.confounders)
            family = params.get("family", "auto")
            return multivariate_regression(
                df,
                target=target,
                features=features,
                family=family,
                hypothesis_id=hypothesis.id,
                alpha=cfg.alpha,
            )

        if method == ProposedMethod.PYTHON_SANDBOX:
            return _execute_sandbox_with_retries(df, hypothesis, profile, cfg, use_llm)

    except Exception as exc:
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            test_used=f"{method.value} (Execution Error)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            error_message=str(exc),
        )

    return ExperimentResult(
        hypothesis_id=hypothesis.id,
        test_used="Unknown Method",
        conclusion=ConclusionStatus.INCONCLUSIVE,
        error_message=f"Unsupported method: {method}",
    )


def _execute_sandbox_with_retries(
    df: pl.DataFrame,
    hypothesis: Hypothesis,
    profile: DatasetProfile | None,
    config: HypoLabConfig,
    use_llm: bool,
) -> ExperimentResult:
    """Run sandbox code and self-heal up to `config.max_sandbox_retries` times on failure."""
    code = str(hypothesis.tool_params.get("code", "")).strip()

    if not code and use_llm and is_llm_configured(config):
        code = _request_sandbox_code_from_llm(hypothesis, profile, config, previous_error=None)

    last_res: ExperimentResult | None = None
    for attempt in range(config.max_sandbox_retries + 1):
        res = run_in_sandbox(
            df,
            code=code,
            hypothesis_id=hypothesis.id,
            timeout_sec=config.sandbox_timeout_sec,
            alpha=config.alpha,
        )
        if res.error_message is None:
            return res

        last_res = res
        if attempt < config.max_sandbox_retries and use_llm and is_llm_configured(config):
            healed = _request_sandbox_code_from_llm(
                hypothesis,
                profile,
                config,
                previous_error=f"Failed code:\n{code}\n\nError:\n{res.error_message}",
            )
            if healed:
                code = healed
                continue
        break

    return last_res or ExperimentResult(
        hypothesis_id=hypothesis.id,
        test_used="Python Sandbox",
        conclusion=ConclusionStatus.INCONCLUSIVE,
        error_message="Failed to generate or execute sandbox script.",
    )


def _request_sandbox_code_from_llm(
    hypothesis: Hypothesis,
    profile: DatasetProfile | None,
    config: HypoLabConfig,
    previous_error: str | None = None,
) -> str:
    """Ask LLM to write or repair Python sandbox code for a hypothesis."""
    try:
        profile_text = profile.to_llm_summary() if profile else f"Variables: {hypothesis.variables}"
        prompt = (
            f"Hypothesis ({hypothesis.id}): {hypothesis.statement}\n"
            f"Null Hypothesis: {hypothesis.null_hypothesis}\n"
            f"Variables: {hypothesis.variables}\n"
            f"Confounders: {hypothesis.confounders}\n\n"
            f"{profile_text}\n"
        )
        if previous_error:
            prompt += f"\nPrevious attempt failed with error — fix the code:\n{previous_error}\n"

        resp = call_llm_json(SANDBOX_SELF_HEAL_PROMPT, prompt, config)
        if isinstance(resp, dict) and "code" in resp:
            return str(resp["code"])
    except Exception:
        pass
    return ""
