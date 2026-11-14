"""Streamlit Interactive Web Application for HypoLab."""

from __future__ import annotations

import json
from typing import Any
import pandas as pd
import plotly.io as pio
import polars as pl
import streamlit as st

from hypolab.agents.orchestrator import run_scientific_investigation
from hypolab.benchmarks import BENCHMARK_DATASETS
from hypolab.config import HypoLabConfig
from hypolab.export.html_report import export_html_report, export_markdown_report
from hypolab.models import ConclusionStatus, ExperimentResult, Hypothesis, ScientificReport

st.set_page_config(
    page_title="HypoLab — Autonomous Data Scientist Agent",
    page_icon="🔬",
    layout="wide",
)


def _render_result_card(res: ExperimentResult, hyp: Hypothesis | None, key_prefix: str) -> None:
    """Render an interactive card for a single hypothesis test result."""
    title = hyp.title if hyp else res.hypothesis_id
    status_emoji = (
        "✅"
        if res.conclusion == ConclusionStatus.VALIDATED
        else ("❌" if res.conclusion == ConclusionStatus.REFUTED else "⚠️")
    )

    with st.container(border=True):
        col_head, col_badge = st.columns([5, 1])
        with col_head:
            st.subheader(f"{status_emoji} [{res.hypothesis_id}] {title}")
        with col_badge:
            st.markdown(f"**Status:** `{res.conclusion.value}`")

        if hyp:
            st.markdown(f"- **Alternative Hypothesis ($H_1$):** {hyp.statement}")
            st.markdown(f"- **Null Hypothesis ($H_0$):** *{hyp.null_hypothesis}*")
            if hyp.confounders:
                st.caption(f"Potential confounders monitored: `{', '.join(hyp.confounders)}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Statistical Test", res.test_used.split("(")[0].strip())
        m2.metric("Raw p-value", f"{res.p_value:.2e}" if res.p_value is not None else "N/A")
        m3.metric(
            "FDR-Adjusted p-value",
            f"{res.adjusted_p_value:.2e}" if res.adjusted_p_value is not None else "N/A",
        )
        m4.metric(
            res.effect_size_name,
            f"{res.effect_size:.3f}" if res.effect_size is not None else "N/A",
        )

        st.markdown(f"**Quantitative Finding:** {res.interpretation}")
        st.info(f"**Critic Verdict:** {res.critic_notes}")

        if res.figure_json:
            try:
                fig = pio.from_json(res.figure_json)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot_{res.hypothesis_id}")
            except Exception:
                pass

        with st.expander("🔍 View Reproducible Python Code & Raw Metrics"):
            st.code(res.code_executed or "# Executed via deterministic statistical wrapper", language="python")
            if res.additional_metrics:
                st.json(res.additional_metrics)


def main() -> None:
    st.title("🔬 HypoLab — Autonomous Data Scientist Agent")
    st.caption(
        "Automated scientific method loop: **Observation & Profiling → Falsifiable Hypothesis Generation ($H_0/H_1$) "
        "→ Hybrid Statistical & Sandbox Testing → Benjamini-Hochberg FDR Critique → Interactive Publication**."
    )

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Laboratory Settings")

        data_source_mode = st.radio(
            "Dataset Source",
            options=["Built-in Benchmark Dataset", "Upload CSV / Parquet"],
            index=0,
        )

        uploaded_file = None
        selected_benchmark = None
        default_goal = "Discover statistically significant relationships, group differences, and confounding structures."

        if data_source_mode == "Built-in Benchmark Dataset":
            selected_benchmark = st.selectbox(
                "Select Benchmark Scenario",
                options=list(BENCHMARK_DATASETS.keys()),
            )
            if selected_benchmark:
                _, default_goal = BENCHMARK_DATASETS[selected_benchmark]
        else:
            uploaded_file = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet", "pq"])

        st.divider()
        st.subheader("🧠 Agent & LLM Engine")
        use_llm = st.toggle("Enable LLM Theorist / Reporter (LiteLLM)", value=True)
        model_name = st.text_input(
            "LiteLLM Model ID",
            value="gemini/gemini-2.5-flash",
            help="Examples: gemini/gemini-2.5-flash, openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022, ollama/llama3.1",
            disabled=not use_llm,
        )
        api_key_input = st.text_input(
            "API Key (optional if set in .env or using Ollama)",
            type="password",
            disabled=not use_llm,
        )

        st.divider()
        st.subheader("📐 Statistical Rigor Controls")
        alpha_val = st.slider(
            "FDR Significance Threshold (α)",
            min_value=0.001,
            max_value=0.10,
            value=0.05,
            step=0.005,
            help="Benjamini-Hochberg False Discovery Rate threshold.",
        )
        min_effect_val = st.slider(
            "Minimum Effect Size Threshold",
            min_value=0.02,
            max_value=0.50,
            value=0.10,
            step=0.02,
            help="Filters out trivial correlations that are only significant due to large sample sizes.",
        )
        max_hyps = st.slider("Max Hypotheses to Test", min_value=3, max_value=10, value=6)

    research_goal = st.text_area(
        "🎯 Research Objective / Scientific Question",
        value=default_goal,
        height=68,
    )

    # Load active dataset preview
    active_df: pl.DataFrame | None = None
    active_name = "dataset.csv"

    if data_source_mode == "Built-in Benchmark Dataset" and selected_benchmark:
        gen_fn, _ = BENCHMARK_DATASETS[selected_benchmark]
        active_df = gen_fn()
        active_name = selected_benchmark.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".csv"
    elif uploaded_file is not None:
        active_name = uploaded_file.name
        if active_name.lower().endswith((".parquet", ".pq")):
            active_df = pl.read_parquet(uploaded_file)
        else:
            active_df = pl.read_csv(uploaded_file, try_parse_dates=True, ignore_errors=True)

    if active_df is not None:
        with st.expander(f"📊 Dataset Preview: `{active_name}` ({active_df.height:,} rows × {active_df.width} columns)", expanded=False):
            st.dataframe(active_df.head(15).to_pandas(), use_container_width=True)

    run_btn = st.button(
        "🚀 Launch Autonomous Scientific Investigation",
        type="primary",
        use_container_width=True,
        disabled=(active_df is None),
    )

    if run_btn and active_df is not None:
        cfg = HypoLabConfig(
            model=model_name,
            alpha=alpha_val,
            min_effect_size=min_effect_val,
            max_hypotheses=max_hyps,
            api_key=api_key_input.strip() or None,
        )

        log_messages: list[str] = []
        with st.status("🔬 Running Autonomous Scientific Method Loop...", expanded=True) as status_box:
            def _ui_callback(stage: str, message: str, _payload: Any) -> None:
                log_messages.append(f"**[{stage.upper()}]** {message}")
                st.write(f"• {message}")

            report = run_scientific_investigation(
                source=active_df,
                research_goal=research_goal,
                dataset_name=active_name,
                config=cfg,
                use_llm=use_llm,
                on_step=_ui_callback,
            )
            status_box.update(label="✅ Scientific Investigation Complete!", state="complete", expanded=False)

        st.session_state["hypolab_report"] = report
        st.session_state["hypolab_logs"] = log_messages

    # Render Report if available in session state
    if "hypolab_report" in st.session_state:
        report: ScientificReport = st.session_state["hypolab_report"]
        prof = report.dataset_profile
        hyp_map = {h.id: h for h in report.hypotheses}

        validated = [r for r in report.results if r.conclusion == ConclusionStatus.VALIDATED]
        refuted = [r for r in report.results if r.conclusion == ConclusionStatus.REFUTED]
        inconclusive = [r for r in report.results if r.conclusion == ConclusionStatus.INCONCLUSIVE]

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Hypotheses Formulated", len(report.hypotheses))
        k2.metric("Validated Discoveries (FDR)", len(validated))
        k3.metric("Refuted Hypotheses", len(refuted))
        k4.metric("Data Quality Alerts", len(prof.quality_alerts))

        tab_exec, tab_val, tab_ref, tab_prof, tab_audit = st.tabs(
            [
                "📑 Executive Synthesis",
                f"✅ Validated Discoveries ({len(validated)})",
                f"❌ Refuted & Controlled ({len(refuted) + len(inconclusive)})",
                "🧬 Dataset Profile & Quality",
                "💻 Audit, Code & Export",
            ]
        )

        with tab_exec:
            st.markdown(report.executive_summary)
            st.subheader("Methodology & Rigor")
            st.write(report.methodology_notes)
            st.subheader("Limitations & Recommended Next Steps")
            for item in report.limitations_and_next_steps:
                st.markdown(f"- {item}")

        with tab_val:
            if not validated:
                st.warning("No hypotheses passed both Benjamini-Hochberg FDR adjustment and minimum effect-size thresholds.")
            else:
                sorted_val = sorted(validated, key=lambda r: abs(r.effect_size or 0.0), reverse=True)
                for res in sorted_val:
                    _render_result_card(res, hyp_map.get(res.hypothesis_id), key_prefix="val")

        with tab_ref:
            st.caption(
                "Scientific transparency requires publishing negative results, false positives caught by Benjamini-Hochberg FDR, "
                "and spurious correlations refuted by confounding controls (such as Simpson's Paradox)."
            )
            others = refuted + inconclusive
            if not others:
                st.info("No hypotheses were refuted in this investigation.")
            else:
                for res in others:
                    _render_result_card(res, hyp_map.get(res.hypothesis_id), key_prefix="ref")

        with tab_prof:
            st.subheader("Column Statistical Profiles")
            col_rows = []
            for c in prof.columns:
                col_rows.append(
                    {
                        "Column": c.name,
                        "Type": c.semantic_type,
                        "Distribution": c.distribution_type.value,
                        "Missing %": c.missing_pct,
                        "Unique": c.unique_count,
                        "Mean": round(c.mean, 3) if c.mean is not None else None,
                        "Std": round(c.std, 3) if c.std is not None else None,
                        "Median": round(c.median, 3) if c.median is not None else None,
                        "Skewness": round(c.skewness, 2) if c.skewness is not None else None,
                        "IQR Outliers": c.outlier_count_iqr,
                    }
                )
            st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

            if prof.top_correlations:
                st.subheader("Top Bivariate Numeric Correlations")
                st.dataframe(
                    pd.DataFrame([c.model_dump() for c in prof.top_correlations]),
                    use_container_width=True,
                )

            if prof.quality_alerts:
                st.subheader("Quality & Structural Alerts")
                for alert in prof.quality_alerts:
                    st.warning(f"**[{alert.severity}]** `{alert.column or 'Dataset'}` — {alert.issue_type}: {alert.description}")

        with tab_audit:
            st.subheader("Export Standalone Reports")
            dl_col1, dl_col2 = st.columns(2)
            html_str = export_html_report(report)
            md_str = export_markdown_report(report)
            with dl_col1:
                st.download_button(
                    "📥 Download Interactive HTML Report (with Plotly)",
                    data=html_str,
                    file_name=f"hypolab_report_{prof.dataset_name}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "📥 Download Markdown Report (.md)",
                    data=md_str,
                    file_name=f"hypolab_report_{prof.dataset_name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.subheader("Complete Reproducible Experiment Log")
            for res in report.results:
                st.markdown(f"**`{res.hypothesis_id}` — {res.test_used} (`{res.conclusion.value}`)**")
                st.code(res.code_executed or "# Deterministic wrapper", language="python")


if __name__ == "__main__":
    main()
