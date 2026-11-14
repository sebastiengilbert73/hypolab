"""Unit tests for AST security validation and isolated subprocess execution in the Python sandbox."""

from __future__ import annotations

import polars as pl

from hypolab.models import ConclusionStatus
from hypolab.tools.sandbox import run_in_sandbox, validate_code_ast


def test_ast_validator_blocks_unsafe_imports_and_builtins() -> None:
    unsafe_snippets = [
        "import os\nos.system('echo hacked')",
        "import subprocess\nsubprocess.run(['dir'])",
        "from pathlib import Path\nPath('secret.txt')",
        "open('secret.txt', 'w')",
        "eval('1 + 1')",
        "__import__('os')",
        "df.__class__.__bases__",
        "df.to_csv('leak.csv')",
    ]
    for code in unsafe_snippets:
        valid, err = validate_code_ast(code)
        assert valid is False, f"Expected snippet to be blocked: {code}"
        assert err is not None


def test_ast_validator_allows_scientific_code_and_executes_in_sandbox() -> None:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.1, 3.9, 6.2, 8.0, 9.9, 12.1, 14.0, 15.8, 18.1, 20.0],
        }
    )

    code = """
r, p_val = stats.pearsonr(df['x'], df['y'])
result = {
    'test_used': 'Custom Pearson Test',
    'statistic_name': 'r',
    'statistic_value': float(r),
    'p_value': float(p_val),
    'effect_size_name': '|r|',
    'effect_size': abs(float(r)),
    'sample_size': len(df),
    'interpretation': f'Strong linear relationship r={r:.3f}'
}
fig = px.scatter(df, x='x', y='y', title='Test Scatter')
"""
    valid, err = validate_code_ast(code)
    assert valid is True, f"Unexpected validation error: {err}"

    res = run_in_sandbox(df, code=code, hypothesis_id="HYP-SBX-1", timeout_sec=15)
    assert res.error_message is None
    assert res.conclusion == ConclusionStatus.VALIDATED
    assert res.p_value is not None and res.p_value < 1e-5
    assert res.figure_json is not None
