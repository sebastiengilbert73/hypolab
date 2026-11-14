"""End-to-end tests for the multi-agent scientific investigation orchestrator,
FDR correction, confounding detection, and HTML/Markdown report export."""

from __future__ import annotations

from hypolab.agents.orchestrator import run_scientific_investigation
from hypolab.benchmarks import (
    generate_nonlinear_and_confounded_dataset,
    generate_saas_churn_dataset,
    generate_simpsons_paradox_dataset,
)
from hypolab.config import HypoLabConfig
from hypolab.export.html_report import export_html_report, export_markdown_report
from hypolab.models import ConclusionStatus


def test_end_to_end_orchestrator_on_saas_churn() -> None:
    df = generate_saas_churn_dataset(n=400, seed=42)
    cfg = HypoLabConfig(alpha=0.05, max_hypotheses=6)
    steps_seen: list[str] = []

    report = run_scientific_investigation(
        source=df,
        research_goal="Identify key drivers of SaaS churn and spending.",
        dataset_name="saas_churn.csv",
        config=cfg,
        use_llm=False,
        on_step=lambda stage, msg, _: steps_seen.append(stage),
    )

    assert "profiler_done" in steps_seen
    assert "theorist_done" in steps_seen
    assert "critic_done" in steps_seen
    assert "reporter_done" in steps_seen

    assert len(report.hypotheses) >= 4
    assert len(report.results) == len(report.hypotheses)
    assert any(r.conclusion == ConclusionStatus.VALIDATED for r in report.results)
    assert all(r.adjusted_p_value is not None for r in report.results if r.p_value is not None)

    html_out = export_html_report(report)
    md_out = export_markdown_report(report)
    assert "Plotly.newPlot" in html_out
    assert "HypoLab Scientific Investigation Report" in md_out


def test_orchestrator_catches_simpsons_paradox() -> None:
    df = generate_simpsons_paradox_dataset(n_per_group=100, seed=42)
    cfg = HypoLabConfig(alpha=0.05, max_hypotheses=5)

    report = run_scientific_investigation(
        source=df,
        research_goal="Test whether experience increases or decreases salary.",
        dataset_name="simpsons_paradox.csv",
        config=cfg,
        use_llm=False,
    )

    # Ensure at least one hypothesis detected Simpson's paradox and was refuted by the Critic
    simpson_notes = [
        r.critic_notes for r in report.results if "Simpson's Paradox" in r.critic_notes
    ]
    assert len(simpson_notes) >= 1


def test_orchestrator_detects_nonlinear_quadratic_effect() -> None:
    df = generate_nonlinear_and_confounded_dataset(n=350, seed=42)
    cfg = HypoLabConfig(alpha=0.05, max_hypotheses=6)

    report = run_scientific_investigation(
        source=df,
        research_goal="Discover non-linear and confounded relationships.",
        dataset_name="nonlinear.csv",
        config=cfg,
        use_llm=False,
    )
    assert len(report.results) >= 3


def test_extract_json_from_text_repairs_latex_backslashes() -> None:
    from hypolab.agents.llm_client import extract_json_from_text

    raw_llm_output = (
        '```json\n'
        '{\n'
        '  "executive_summary": "Significant at \\alpha = 0.05 ($p \\le 10^{-5}$, \\chi^2 = 14.2, \\\\rho = 0.6).",\n'
        '  "methodology_notes": "Tested \\beta_1 \\neq 0.",\n'
        '}\n'
        '```'
    )
    parsed = extract_json_from_text(raw_llm_output)
    assert isinstance(parsed, dict)
    assert "\\alpha" in parsed["executive_summary"]
    assert "\\rho" in parsed["executive_summary"]
