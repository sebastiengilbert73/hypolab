# 🔬 HypoLab

**Autonomous Data Scientist Agent for Statistical Hypothesis Generation and Testing.**

Unlike traditional *"Chat with your CSV"* tools that merely translate user prompts into SQL or Pandas snippets, **HypoLab** operates as an autonomous quantitative researcher. Given a dataset (CSV, Parquet, or built-in benchmark) and a research goal, it executes a complete **5-stage scientific method loop**:

1. **Observation & Profiling (`hypolab.profiler`)**: Ultra-fast Polars profiling of schema, distribution shapes (normal, skewed, multimodal, binary, categorical), missingness, IQR outliers, and pairwise Pearson/Spearman correlations.
2. **Hypothesis Formulation (`hypolab.agents.theorist`)**: Formulates falsifiable hypotheses ($H_0$ vs $H_1$) with explicit explanatory variables, target outcomes, and potential confounders using LiteLLM (OpenAI, Gemini, Anthropic, or local Ollama) with a deterministic heuristic fallback.
3. **Hybrid Experimentation (`hypolab.tools` & `hypolab.agents.experimenter`)**:
   - **Deterministic Statistical Suite**: Automatically checks Shapiro-Wilk normality and Levene homoscedasticity to choose between **Student's $t$-test**, **Welch's $t$-test**, **Mann-Whitney U**, **One-Way ANOVA**, **Kruskal-Wallis + Tukey HSD**, **Pearson's $\chi^2$**, **Fisher's Exact Test**, and **Multivariate OLS / Logistic Regression**.
   - **Isolated Python Sandbox**: AST-validated subprocess runner (blocking file/network/system access) with automatic self-healing retries for custom feature engineering and polynomial/non-linear modeling.
4. **Statistical Critique & Rigor Control (`hypolab.agents.critic`)**:
   - Applies **Benjamini-Hochberg False Discovery Rate (FDR)** correction across all tested hypotheses to prevent *p-hacking*.
   - Enforces practical significance via minimum **effect size** thresholds (Cohen's $d$, Rank-Biserial $r$, $\eta^2$, $\epsilon^2$, Cramér's $V$, Adjusted $R^2$).
   - Detects **Simpson's Paradox** sign reversals and runs multivariate confounding checks.
5. **Interactive Publication (`app/main.py` & `hypolab.export`)**: Streamlit web application with live laboratory logs, interactive Plotly charts, refuted hypothesis transparency, reproducible Python code for every test, and standalone HTML/Markdown report export.

---

## Quick Start

### 1. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Launch the Interactive Web UI (Streamlit)

```powershell
.\.venv\Scripts\streamlit.exe run app/main.py
```
Or simply run:
```powershell
.\hypolab.bat ui
```

### 3. Command-Line Interface (Rich CLI)

Run an autonomous investigation on one of the built-in synthetic benchmark datasets (`churn`, `simpson`, `nonlinear`) or your own CSV/Parquet file:

```powershell
# Run on the SaaS Churn benchmark and export an interactive HTML report
.\.venv\Scripts\hypolab.exe --demo churn --no-llm --export-html churn_report.html

# Run on the Simpson's Paradox benchmark
.\.venv\Scripts\hypolab.exe --demo simpson --no-llm

# Run on a custom CSV file with Gemini / OpenAI / Ollama
.\.venv\Scripts\hypolab.exe my_data.csv --goal "Analyze key drivers of customer retention" --model gemini/gemini-3.8-flash
```

---

## Project Architecture

```text
hypolab/
├── pyproject.toml
├── README.md
├── .env.example
├── hypolab.bat                  # Convenience launcher for Windows (UI & CLI)
├── app/
│   └── main.py                  # Interactive Streamlit Web UI
├── data/
│   ├── saas_churn_benchmark.csv           # Multi-factor SaaS customer churn dataset
│   ├── simpsons_paradox_benchmark.csv     # Simpson's Paradox (salary vs experience by dept)
│   └── nonlinear_confounded_benchmark.csv # Quadratic U-curve & temperature confounder dataset
├── hypolab/
│   ├── __init__.py
│   ├── config.py                # Global configuration (LiteLLM, FDR alpha, timeouts)
│   ├── models.py                # Strict Pydantic v2 schemas
│   ├── benchmarks.py            # Synthetic benchmark datasets (Simpson's Paradox, Churn, Non-Linear)
│   ├── cli.py                   # Rich terminal CLI
│   ├── profiler/
│   │   ├── engine.py            # Fast Polars/SciPy dataset profiler
│   │   └── quality.py           # Outlier, missingness, and multicollinearity detector
│   ├── tools/
│   │   ├── registry.py          # Tool schema registry
│   │   ├── statistical_tests.py # Parametric/non-parametric tests, effect sizes & Plotly charts
│   │   └── sandbox.py           # AST security validator & isolated subprocess runner
│   ├── agents/
│   │   ├── llm_client.py        # LiteLLM structured JSON completion wrapper
│   │   ├── theorist.py          # Hypothesis formulation (LLM + deterministic heuristic engine)
│   │   ├── experimenter.py      # Tool dispatcher & self-healing sandbox executor
│   │   ├── critic.py            # Benjamini-Hochberg FDR correction & confounding control
│   │   ├── reporter.py          # Executive synthesis & methodology narrator
│   │   └── orchestrator.py      # 5-stage scientific loop coordinator
│   └── export/
│       └── html_report.py       # Standalone HTML (Plotly.js) and Markdown exporter
└── tests/
    ├── test_profiler.py
    ├── test_statistical_tools.py
    ├── test_sandbox_security.py
    └── test_orchestrator.py
```

---

## Running Tests

```powershell
.\.venv\Scripts\pytest.exe -v
```
