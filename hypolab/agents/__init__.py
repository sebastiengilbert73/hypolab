"""Multi-agent scientific investigation modules."""

from hypolab.agents.critic import review_and_adjust_results
from hypolab.agents.experimenter import run_experiment
from hypolab.agents.orchestrator import run_scientific_investigation
from hypolab.agents.reporter import build_scientific_report
from hypolab.agents.theorist import generate_heuristic_hypotheses, generate_hypotheses

__all__ = [
    "run_scientific_investigation",
    "generate_hypotheses",
    "generate_heuristic_hypotheses",
    "run_experiment",
    "review_and_adjust_results",
    "build_scientific_report",
]
