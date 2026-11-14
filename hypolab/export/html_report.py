"""Export a ScientificReport to standalone interactive HTML (with Plotly.js) or Markdown."""

from __future__ import annotations

import html
import json
from hypolab.models import ConclusionStatus, ScientificReport


def export_markdown_report(report: ScientificReport) -> str:
    """Render the ScientificReport as a structured Markdown document."""
    prof = report.dataset_profile
    hyp_map = {h.id: h for h in report.hypotheses}

    validated = [r for r in report.results if r.conclusion == ConclusionStatus.VALIDATED]
    refuted = [r for r in report.results if r.conclusion == ConclusionStatus.REFUTED]

    lines = [
        f"# HypoLab Scientific Investigation Report: {prof.dataset_name}",
        "",
        f"- **Research Goal:** {report.research_goal}",
        f"- **Dataset Dimensions:** {prof.row_count:,} rows × {prof.column_count} columns (Duplicates: {prof.duplicate_row_count})",
        f"- **Significance Threshold (FDR α):** {report.alpha}",
        f"- **Hypotheses Tested:** {len(report.hypotheses)} ({len(validated)} Validated, {len(refuted)} Refuted)",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        report.executive_summary,
        "",
        "---",
        "",
        "## 2. Validated Discoveries",
        "",
    ]

    if not validated:
        lines.append("*No hypotheses passed both Benjamini-Hochberg FDR correction and practical effect-size thresholds.*")
    else:
        for res in validated:
            hyp = hyp_map.get(res.hypothesis_id)
            title = hyp.title if hyp else res.hypothesis_id
            lines.extend(
                [
                    f"### [{res.hypothesis_id}] {title}",
                    f"- **Alternative Hypothesis ($H_1$):** {hyp.statement if hyp else ''}",
                    f"- **Null Hypothesis ($H_0$):** {hyp.null_hypothesis if hyp else ''}",
                    f"- **Statistical Test:** `{res.test_used}` ({res.statistic_name} = {res.statistic_value})",
                    f"- **Raw $p$-value:** `{res.p_value:.3e}` | **FDR-Adjusted $p$-value:** `{res.adjusted_p_value:.3e}`",
                    f"- **Effect Size:** {res.effect_size_name} = `{res.effect_size}` (N = {res.sample_size})",
                    f"- **Interpretation:** {res.interpretation}",
                    f"- **Critic Verdict:** {res.critic_notes}",
                    "",
                    "```python",
                    res.code_executed,
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "---",
            "",
            "## 3. Refuted Hypotheses (Scientific Transparency)",
            "",
        ]
    )
    if not refuted:
        lines.append("*No hypotheses were refuted in this run.*")
    else:
        for res in refuted:
            hyp = hyp_map.get(res.hypothesis_id)
            title = hyp.title if hyp else res.hypothesis_id
            p_str = f"{res.p_value:.3e}" if res.p_value is not None else "N/A"
            adj_str = f"{res.adjusted_p_value:.3e}" if res.adjusted_p_value is not None else "N/A"
            lines.extend(
                [
                    f"### [{res.hypothesis_id}] {title} — REFUTED",
                    f"- **Statement ($H_1$):** {hyp.statement if hyp else ''}",
                    f"- **Test Used:** `{res.test_used}` | Raw $p$ = `{p_str}` | FDR-Adjusted $p$ = `{adj_str}` | {res.effect_size_name} = `{res.effect_size}`",
                    f"- **Critic Rejection Reason:** {res.critic_notes}",
                    "",
                ]
            )

    lines.extend(
        [
            "---",
            "",
            "## 4. Methodology & Limitations",
            "",
            report.methodology_notes,
            "",
            "### Limitations & Next Steps",
        ]
    )
    for lim in report.limitations_and_next_steps:
        lines.append(f"- {lim}")

    return "\n".join(lines)


def export_html_report(report: ScientificReport) -> str:
    """Render a standalone HTML report with embedded interactive Plotly charts."""
    prof = report.dataset_profile
    hyp_map = {h.id: h for h in report.hypotheses}

    cards_html: list[str] = []
    for idx, res in enumerate(report.results):
        hyp = hyp_map.get(res.hypothesis_id)
        title = html.escape(hyp.title if hyp else res.hypothesis_id)
        stmt = html.escape(hyp.statement if hyp else "")
        h0 = html.escape(hyp.null_hypothesis if hyp else "")
        badge_color = (
            "#16a34a"
            if res.conclusion == ConclusionStatus.VALIDATED
            else ("#dc2626" if res.conclusion == ConclusionStatus.REFUTED else "#d97706")
        )
        p_str = f"{res.p_value:.3e}" if res.p_value is not None else "N/A"
        adj_str = f"{res.adjusted_p_value:.3e}" if res.adjusted_p_value is not None else "N/A"
        eff_str = f"{res.effect_size:.4f}" if res.effect_size is not None else "N/A"

        chart_div = ""
        if res.figure_json:
            div_id = f"plotly_chart_{idx}"
            # Safely serialize JSON for inline script
            fig_dict = json.loads(res.figure_json)
            data_json = json.dumps(fig_dict.get("data", []))
            layout_json = json.dumps(fig_dict.get("layout", {}))
            chart_div = f"""
            <div id="{div_id}" style="width:100%;height:420px;margin-top:12px;"></div>
            <script>
                Plotly.newPlot("{div_id}", {data_json}, {layout_json}, {{responsive: true}});
            </script>
            """

        code_block = html.escape(res.code_executed or "")
        cards_html.append(
            f"""
            <div class="card">
                <div class="card-header">
                    <h3>[{html.escape(res.hypothesis_id)}] {title}</h3>
                    <span class="badge" style="background:{badge_color};">{res.conclusion.value}</span>
                </div>
                <p><strong>H₁:</strong> {stmt}</p>
                <p><strong>H₀:</strong> {h0}</p>
                <div class="metrics">
                    <span><strong>Test:</strong> {html.escape(res.test_used)}</span>
                    <span><strong>Raw p:</strong> {p_str}</span>
                    <span><strong>FDR p:</strong> {adj_str}</span>
                    <span><strong>{html.escape(res.effect_size_name)}:</strong> {eff_str}</span>
                    <span><strong>N:</strong> {res.sample_size}</span>
                </div>
                <p><strong>Interpretation:</strong> {html.escape(res.interpretation)}</p>
                <p class="critic"><strong>Critic Assessment:</strong> {html.escape(res.critic_notes)}</p>
                {chart_div}
                <details>
                    <summary>Reproducible Python Code</summary>
                    <pre><code>{code_block}</code></pre>
                </details>
            </div>
            """
        )

    exec_html = html.escape(report.executive_summary).replace("\n", "<br>")
    meth_html = html.escape(report.methodology_notes)
    lims_html = "".join(f"<li>{html.escape(item)}</li>" for item in report.limitations_and_next_steps)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HypoLab Report — {html.escape(prof.dataset_name)}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; max-width: 1100px; margin: 0 auto; padding: 32px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #1e293b; }}
        .header-box {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 12px; }}
        .card-header h3 {{ margin: 0; }}
        .badge {{ color: #fff; font-weight: 700; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 16px; background: #f1f5f9; padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; margin: 12px 0; }}
        .critic {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 4px; }}
        pre {{ background: #0f172a; color: #f8fafc; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }}
        details {{ margin-top: 12px; }}
        summary {{ cursor: pointer; font-weight: 600; color: #2563eb; }}
    </style>
</head>
<body>
    <div class="header-box">
        <h1>🔬 HypoLab Scientific Report: {html.escape(prof.dataset_name)}</h1>
        <p><strong>Research Goal:</strong> {html.escape(report.research_goal)}</p>
        <p><strong>Dataset Size:</strong> {prof.row_count:,} rows × {prof.column_count} columns | <strong>FDR Significance (α):</strong> {report.alpha}</p>
    </div>

    <div class="header-box">
        <h2>Executive Summary</h2>
        <p>{exec_html}</p>
    </div>

    <h2>Experimental Hypotheses & Results</h2>
    {"".join(cards_html)}

    <div class="header-box">
        <h2>Methodology & Limitations</h2>
        <p>{meth_html}</p>
        <ul>{lims_html}</ul>
    </div>
</body>
</html>
"""
