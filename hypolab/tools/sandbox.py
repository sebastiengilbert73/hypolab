"""Isolated Python Sandbox with AST security validation and timeout enforcement."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from hypolab.models import ConclusionStatus, ExperimentResult

ALLOWED_IMPORT_ROOTS = {
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "statsmodels",
    "sklearn",
    "plotly",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "re",
}

FORBIDDEN_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
}

FORBIDDEN_ATTRIBUTES = {
    "__builtins__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__dict__",
    "__class__",
    "to_csv",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "read_csv",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_json",
    "write_csv",
    "write_parquet",
}


class _SandboxASTVisitor(ast.NodeVisitor):
    """AST visitor that enforces module import whitelist and blocks unsafe calls/attributes."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.violations.append(f"Disallowed import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.violations.append("Relative imports are disallowed in sandbox.")
        else:
            root = node.module.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.violations.append(f"Disallowed import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.violations.append(f"Disallowed builtin call: '{node.func.id}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("__"):
            self.violations.append(f"Disallowed attribute access: '{node.attr}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.violations.append(f"Disallowed dunder identifier: '{node.id}'")
        self.generic_visit(node)


def validate_code_ast(code: str) -> tuple[bool, str | None]:
    """Validate Python code string against AST security rules."""
    if not code or not code.strip():
        return False, "Empty code string."

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} at line {exc.lineno}"

    visitor = _SandboxASTVisitor()
    visitor.visit(tree)
    if visitor.violations:
        return False, "; ".join(visitor.violations)

    return True, None


_RUNNER_TEMPLATE = """\
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

pl_df = pl.read_parquet({parquet_path!r})
df = pl_df.to_pandas()

result = {{}}
fig = None

# --- User / Agent Code Start ---
{user_code}
# --- User / Agent Code End ---

output_payload = {{
    "result": result if isinstance(result, dict) else {{"raw": str(result)}},
    "figure_json": fig.to_json() if (fig is not None and hasattr(fig, "to_json")) else None,
}}

with open({output_path!r}, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, default=str)
"""


def run_in_sandbox(
    df: pl.DataFrame | pd.DataFrame,
    code: str,
    hypothesis_id: str = "HYP-000",
    timeout_sec: int = 15,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Validate and execute custom statistical Python code in an isolated subprocess."""
    is_valid, err_msg = validate_code_ast(code)
    if not is_valid:
        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used="Python Sandbox (Blocked by AST Validator)",
            conclusion=ConclusionStatus.INCONCLUSIVE,
            code_executed=code,
            error_message=f"Security/Syntax Validation Failed: {err_msg}",
        )

    pl_df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)

    with tempfile.TemporaryDirectory(prefix="hypolab_sbx_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        parquet_file = tmp_path / "data.parquet"
        script_file = tmp_path / "runner.py"
        output_file = tmp_path / "output.json"

        pl_df.write_parquet(parquet_file)
        runner_src = _RUNNER_TEMPLATE.format(
            parquet_path=str(parquet_file),
            output_path=str(output_file),
            user_code=code,
        )
        script_file.write_text(runner_src, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_file)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(tmp_path),
            )
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                hypothesis_id=hypothesis_id,
                test_used="Python Sandbox (Timeout)",
                conclusion=ConclusionStatus.INCONCLUSIVE,
                code_executed=code,
                error_message=f"Execution timed out after {timeout_sec} seconds.",
            )

        if proc.returncode != 0:
            stderr_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown runtime error"
            return ExperimentResult(
                hypothesis_id=hypothesis_id,
                test_used="Python Sandbox (Runtime Error)",
                conclusion=ConclusionStatus.INCONCLUSIVE,
                code_executed=code,
                error_message=stderr_msg[-1500:],
            )

        if not output_file.exists():
            return ExperimentResult(
                hypothesis_id=hypothesis_id,
                test_used="Python Sandbox (No Output)",
                conclusion=ConclusionStatus.INCONCLUSIVE,
                code_executed=code,
                error_message="Sandbox exited cleanly but produced no output payload.",
            )

        payload: dict[str, Any] = json.loads(output_file.read_text(encoding="utf-8"))
        res_dict: dict[str, Any] = payload.get("result", {})
        fig_json: str | None = payload.get("figure_json")

        test_used = str(res_dict.get("test_used", "Custom Sandbox Statistical Test"))
        stat_name = str(res_dict.get("statistic_name", "stat"))
        stat_val = _safe_float(res_dict.get("statistic_value"))
        p_val = _safe_float(res_dict.get("p_value"))
        eff_name = str(res_dict.get("effect_size_name", "effect_size"))
        eff_val = _safe_float(res_dict.get("effect_size"))
        sample_size = int(res_dict.get("sample_size", pl_df.height))
        interp = str(res_dict.get("interpretation", proc.stdout.strip() or "Sandbox test completed."))

        if p_val is not None:
            eff_magnitude = abs(eff_val) if eff_val is not None else 1.0
            conclusion = (
                ConclusionStatus.VALIDATED
                if (p_val < alpha and eff_magnitude >= 0.05)
                else ConclusionStatus.REFUTED
            )
        else:
            conclusion = ConclusionStatus.INCONCLUSIVE

        return ExperimentResult(
            hypothesis_id=hypothesis_id,
            test_used=test_used,
            statistic_name=stat_name,
            statistic_value=stat_val,
            p_value=p_val,
            effect_size_name=eff_name,
            effect_size=eff_val,
            sample_size=sample_size,
            conclusion=conclusion,
            interpretation=interp,
            code_executed=code,
            additional_metrics={
                k: v
                for k, v in res_dict.items()
                if k
                not in {
                    "test_used",
                    "statistic_name",
                    "statistic_value",
                    "p_value",
                    "effect_size_name",
                    "effect_size",
                    "sample_size",
                    "interpretation",
                }
            },
            figure_json=fig_json,
        )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return round(f, 6)
    except (ValueError, TypeError):
        return None
