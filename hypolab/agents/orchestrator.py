"""Orchestrator Agent: Runs the complete 5-stage autonomous scientific method loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import pandas as pd
import polars as pl

from hypolab.agents.critic import review_and_adjust_results
from hypolab.agents.experimenter import run_experiment
from hypolab.agents.reporter import build_scientific_report
from hypolab.agents.theorist import generate_hypotheses
from hypolab.config import HypoLabConfig
from hypolab.models import ExperimentResult, ScientificReport
from hypolab.profiler.engine import load_dataset, profile_dataset

StepCallback = Callable[[str, str, Any], None]


def run_scientific_investigation(
    source: str | Path | pl.DataFrame | pd.DataFrame,
    research_goal: str = "Discover statistically significant relationships, group differences, and confounding structures.",
    dataset_name: str | None = None,
    config: HypoLabConfig | None = None,
    use_llm: bool = True,
    on_step: StepCallback | None = None,
) -> ScientificReport:
    """Execute the full HypoLab autonomous research loop:
    1. Observation & Profiling
    2. Hypothesis Formulation (Theorist)
    3. Hybrid Experimentation (Experimenter: Structured Tools + Sandbox)
    4. Statistical Critique & FDR Control (Critic)
    5. Scientific Publication (Reporter)
    """
    cfg = config or HypoLabConfig()

    def _notify(stage: str, message: str, payload: Any = None) -> None:
        if on_step is not None:
            on_step(stage, message, payload)

    # Stage 1: Load & Profile Dataset
    _notify("profiler", "Profiling dataset schema, distributions, outliers, and correlations...")
    df, inferred_name = load_dataset(source)
    name = dataset_name or inferred_name
    profile = profile_dataset(df, dataset_name=name)
    _notify(
        "profiler_done",
        f"Profiled {profile.row_count:,} rows x {profile.column_count} columns ({len(profile.quality_alerts)} quality alerts).",
        profile,
    )

    # Stage 2: Formulate Falsifiable Hypotheses
    _notify("theorist", "Formulating falsifiable hypotheses (H0 vs H1) and identifying potential confounders...")
    hypotheses = generate_hypotheses(
        profile=profile,
        research_goal=research_goal,
        config=cfg,
        use_llm=use_llm,
    )
    _notify(
        "theorist_done",
        f"Formulated {len(hypotheses)} falsifiable hypotheses.",
        hypotheses,
    )

    # Stage 3: Execute Experiments
    raw_results: list[ExperimentResult] = []
    for idx, hyp in enumerate(hypotheses, start=1):
        _notify(
            "experimenter",
            f"Testing [{hyp.id}] ({idx}/{len(hypotheses)}): {hyp.title} via `{hyp.proposed_method.value}`...",
            hyp,
        )
        res = run_experiment(
            df=df,
            hypothesis=hyp,
            profile=profile,
            config=cfg,
            use_llm=use_llm,
        )
        raw_results.append(res)
        _notify("experimenter_step", f"Completed [{hyp.id}]: {res.test_used} (raw p={res.p_value})", res)

    # Stage 4: Critique, FDR Adjustment & Confounding Control
    _notify("critic", "Applying Benjamini-Hochberg FDR correction, effect size filters, and confounding controls...")
    reviewed_results = review_and_adjust_results(
        df=df,
        hypotheses=hypotheses,
        results=raw_results,
        config=cfg,
    )
    _notify("critic_done", "Completed statistical critique and multiple-testing adjustment.", reviewed_results)

    # Stage 5: Synthesize Final Scientific Report
    _notify("reporter", "Synthesizing final scientific report and visualizations...")
    report = build_scientific_report(
        profile=profile,
        research_goal=research_goal,
        hypotheses=hypotheses,
        results=reviewed_results,
        config=cfg,
        use_llm=use_llm,
    )
    _notify("reporter_done", "Scientific report ready.", report)
    return report
