"""Reporter Agent: Synthesizes validated findings, refuted hypotheses, and methodology notes."""

from __future__ import annotations

from hypolab.agents.llm_client import call_llm_json, is_llm_configured
from hypolab.config import HypoLabConfig
from hypolab.models import (
    ConclusionStatus,
    DatasetProfile,
    ExperimentResult,
    Hypothesis,
    ScientificReport,
)

REPORTER_SYSTEM_PROMPT = """\
You are the Lead Principal Investigator (Reporter Agent) in HypoLab.
Synthesize the results of an autonomous scientific investigation into a clear, rigorous executive report.

Return ONLY a JSON object with keys:
- "executive_summary": 2-3 paragraphs summarizing key validated discoveries, their effect sizes, and important refuted hypotheses (including any confounding or Simpson's paradox findings).
- "methodology_notes": 1 paragraph describing the profiling, parametric/non-parametric test selection, Benjamini-Hochberg FDR correction, and confounding control.
- "limitations_and_next_steps": List of 3-4 actionable follow-up recommendations or limitations.
"""


def build_scientific_report(
    profile: DatasetProfile,
    research_goal: str,
    hypotheses: list[Hypothesis],
    results: list[ExperimentResult],
    config: HypoLabConfig | None = None,
    use_llm: bool = True,
) -> ScientificReport:
    """Create the final ScientificReport combining all hypotheses, results, and narrative synthesis."""
    cfg = config or HypoLabConfig()
    hyp_map = {h.id: h for h in hypotheses}

    validated = [r for r in results if r.conclusion == ConclusionStatus.VALIDATED]
    refuted = [r for r in results if r.conclusion == ConclusionStatus.REFUTED]
    inconclusive = [r for r in results if r.conclusion == ConclusionStatus.INCONCLUSIVE]

    if use_llm and is_llm_configured(cfg):
        try:
            findings_lines = []
            for r in results:
                h = hyp_map.get(r.hypothesis_id)
                title = h.title if h else r.hypothesis_id
                findings_lines.append(
                    f"- [{r.conclusion.value}] {r.hypothesis_id} ({title}): {r.test_used}, "
                    f"raw_p={r.p_value}, fdr_p={r.adjusted_p_value}, "
                    f"{r.effect_size_name}={r.effect_size}. Critic: {r.critic_notes}"
                )
            user_prompt = (
                f"Dataset: {profile.dataset_name} ({profile.row_count} rows, {profile.column_count} cols)\n"
                f"Research Goal: {research_goal}\n"
                f"Significance Level (alpha): {cfg.alpha}\n\n"
                f"Experimental Findings:\n" + "\n".join(findings_lines)
            )
            raw = call_llm_json(REPORTER_SYSTEM_PROMPT, user_prompt, cfg)
            if isinstance(raw, dict) and "executive_summary" in raw:
                return ScientificReport(
                    dataset_profile=profile,
                    research_goal=research_goal,
                    alpha=cfg.alpha,
                    hypotheses=hypotheses,
                    results=results,
                    executive_summary=str(raw["executive_summary"]),
                    methodology_notes=str(raw.get("methodology_notes", _default_methodology(cfg))),
                    limitations_and_next_steps=[
                        str(x) for x in raw.get("limitations_and_next_steps", _default_limitations(profile))
                    ],
                )
        except Exception:
            pass

    # Deterministic structured synthesis
    exec_lines = [
        f"An autonomous quantitative investigation was conducted on **{profile.dataset_name}** "
        f"({profile.row_count:,} observations across {profile.column_count} variables) "
        f"with the objective: *{research_goal}*. Out of **{len(hypotheses)}** falsifiable hypotheses formulated "
        f"and tested, **{len(validated)}** were statistically validated (at FDR-adjusted α = {cfg.alpha}), "
        f"**{len(refuted)}** were refuted, and **{len(inconclusive)}** were inconclusive."
    ]

    if validated:
        top_val = sorted(validated, key=lambda r: abs(r.effect_size or 0.0), reverse=True)
        bullet_items = []
        for r in top_val:
            h = hyp_map.get(r.hypothesis_id)
            title = h.title if h else r.hypothesis_id
            adj_p_str = f"{r.adjusted_p_value:.2e}" if r.adjusted_p_value is not None else "N/A"
            bullet_items.append(
                f"**{r.hypothesis_id} — {title}**: Validated via *{r.test_used}* "
                f"(FDR $p = {adj_p_str}$, {r.effect_size_name} = {r.effect_size:.3f}). {r.interpretation}"
            )
        exec_lines.append("### Key Validated Discoveries\n" + "\n".join(f"- {b}" for b in bullet_items))

    if refuted:
        ref_items = []
        for r in refuted:
            h = hyp_map.get(r.hypothesis_id)
            title = h.title if h else r.hypothesis_id
            ref_items.append(f"**{r.hypothesis_id} — {title}**: {r.critic_notes}")
        exec_lines.append("### Refuted Hypotheses & Confounding Controls\n" + "\n".join(f"- {b}" for b in ref_items))

    return ScientificReport(
        dataset_profile=profile,
        research_goal=research_goal,
        alpha=cfg.alpha,
        hypotheses=hypotheses,
        results=results,
        executive_summary="\n\n".join(exec_lines),
        methodology_notes=_default_methodology(cfg),
        limitations_and_next_steps=_default_limitations(profile),
    )


def _default_methodology(config: HypoLabConfig) -> str:
    return (
        f"All hypotheses were evaluated using a hybrid statistical engine combining deterministic parametric/non-parametric "
        f"tests (Shapiro-Wilk normality and Levene homoscedasticity routing) with an AST-validated isolated Python sandbox. "
        f"To prevent p-hacking across multiple comparisons, p-values were adjusted using the Benjamini-Hochberg False Discovery "
        f"Rate (FDR) procedure at α = {config.alpha}, paired with a minimum practical effect-size threshold "
        f"(≥ {config.min_effect_size}) and multivariate regression checks for confounding variables."
    )


def _default_limitations(profile: DatasetProfile) -> list[str]:
    lims = [
        "Observational statistical associations and regression controls reduce confounding risk but do not establish randomized causal proof.",
        "Non-linear interactions involving three or more higher-order terms could be explored further using tree-based SHAP analysis.",
    ]
    if profile.row_count < 100:
        lims.append(
            f"Sample size is relatively modest (N={profile.row_count}); subtle effects may require larger sample collection to achieve 80% statistical power."
        )
    if profile.quality_alerts:
        lims.append(
            f"Dataset triggered {len(profile.quality_alerts)} data quality alert(s) during profiling which should be reviewed prior to production deployment."
        )
    return lims
