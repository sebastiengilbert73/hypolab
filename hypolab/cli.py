"""Command-line interface for HypoLab using Rich."""

from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypolab.agents.orchestrator import run_scientific_investigation
from hypolab.benchmarks import (
    generate_nonlinear_and_confounded_dataset,
    generate_saas_churn_dataset,
    generate_simpsons_paradox_dataset,
)
from hypolab.config import HypoLabConfig
from hypolab.export.html_report import export_html_report, export_markdown_report
from hypolab.models import ConclusionStatus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypoLab: Autonomous Data Scientist Agent for hypothesis generation and testing."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to CSV/Parquet dataset (omit if using --demo)",
    )
    parser.add_argument(
        "--demo",
        choices=["churn", "simpson", "nonlinear"],
        default="churn",
        help="Run on a built-in synthetic benchmark dataset if no file is provided.",
    )
    parser.add_argument(
        "--goal",
        default="Discover statistically significant relationships, group differences, and confounding structures.",
        help="Research objective for the agent.",
    )
    parser.add_argument("--model", default=None, help="LiteLLM model identifier (e.g. gemini/gemini-2.5-flash, ollama/llama3.1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance level alpha (default: 0.05)")
    parser.add_argument("--max-hypotheses", type=int, default=6, help="Maximum hypotheses to test")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic heuristic theorist instead of LLM")
    parser.add_argument("--export-html", type=str, default=None, help="Path to save standalone HTML report")
    parser.add_argument("--export-md", type=str, default=None, help="Path to save Markdown report")

    args = parser.parse_args()
    console = Console()

    config = HypoLabConfig(alpha=args.alpha, max_hypotheses=args.max_hypotheses)
    if args.model:
        config.model = args.model

    if args.dataset:
        source = args.dataset
        name = Path(args.dataset).name
    else:
        if args.demo == "simpson":
            source = generate_simpsons_paradox_dataset()
            name = "simpsons_paradox_benchmark.csv"
        elif args.demo == "nonlinear":
            source = generate_nonlinear_and_confounded_dataset()
            name = "nonlinear_confounded_benchmark.csv"
        else:
            source = generate_saas_churn_dataset()
            name = "saas_churn_benchmark.csv"

    console.print(Panel.fit(f"[bold cyan]🔬 HypoLab Autonomous Data Scientist[/bold cyan]\nDataset: [bold]{name}[/bold]", border_style="cyan"))

    def _step_logger(stage: str, msg: str, _payload: object) -> None:
        console.print(f"[dim]• [{stage}][/dim] {msg}")

    report = run_scientific_investigation(
        source=source,
        research_goal=args.goal,
        dataset_name=name,
        config=config,
        use_llm=not args.no_llm,
        on_step=_step_logger,
    )

    table = Table(title=f"HypoLab Experimental Results (FDR α = {config.alpha})", show_lines=True)
    table.add_column("ID", style="bold")
    table.add_column("Hypothesis Title")
    table.add_column("Test Used")
    table.add_column("Raw p", justify="right")
    table.add_column("FDR Adj. p", justify="right")
    table.add_column("Effect Size", justify="right")
    table.add_column("Status", justify="center")

    hyp_map = {h.id: h for h in report.hypotheses}
    for res in report.results:
        hyp = hyp_map.get(res.hypothesis_id)
        title = hyp.title if hyp else res.hypothesis_id
        raw_p = f"{res.p_value:.2e}" if res.p_value is not None else "N/A"
        adj_p = f"{res.adjusted_p_value:.2e}" if res.adjusted_p_value is not None else "N/A"
        eff = f"{res.effect_size_name}={res.effect_size:.3f}" if res.effect_size is not None else "N/A"
        if res.conclusion == ConclusionStatus.VALIDATED:
            badge = "[bold green]VALIDATED[/bold green]"
        elif res.conclusion == ConclusionStatus.REFUTED:
            badge = "[bold red]REFUTED[/bold red]"
        else:
            badge = "[bold yellow]INCONCLUSIVE[/bold yellow]"
        table.add_row(res.hypothesis_id, title, res.test_used, raw_p, adj_p, eff, badge)

    console.print()
    console.print(table)

    if args.export_html:
        Path(args.export_html).write_text(export_html_report(report), encoding="utf-8")
        console.print(f"[green]✔ Saved interactive HTML report to {args.export_html}[/green]")

    if args.export_md:
        Path(args.export_md).write_text(export_markdown_report(report), encoding="utf-8")
        console.print(f"[green]✔ Saved Markdown report to {args.export_md}[/green]")


if __name__ == "__main__":
    main()
