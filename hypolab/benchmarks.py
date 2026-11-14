"""Synthetic benchmark datasets featuring classic statistical traps:
1. Simpson's Paradox (Compensation vs Experience confounded by Department)
2. SaaS Customer Churn (Multi-factor behavioral + contract drivers + spurious noise)
3. Non-Linear U-Shaped Dose-Response & Spurious Confounding
"""

from __future__ import annotations

import numpy as np
import polars as pl


def generate_simpsons_paradox_dataset(n_per_group: int = 150, seed: int = 42) -> pl.DataFrame:
    """Generate a dataset exhibiting Simpson's Paradox:
    Aggregate correlation between `years_experience` and `salary_k` is negative,
    but within every single `department` ('Junior_Support', 'Mid_Product', 'Senior_AI_Lab'),
    experience is strongly positively correlated with salary!
    """
    rng = np.random.default_rng(seed)
    depts = [
        ("Support_Ops", 8.0, 55.0),
        ("Product_Eng", 5.0, 95.0),
        ("AI_Research", 2.0, 140.0),
    ]

    rows_dept: list[str] = []
    rows_exp: list[float] = []
    rows_sal: list[float] = []
    rows_remote: list[str] = []
    rows_perf: list[float] = []

    for name, mean_exp, base_sal in depts:
        exp = np.clip(rng.normal(mean_exp, 1.2, size=n_per_group), 0.5, 15.0)
        # Positive slope (+4.5k per year) within each department!
        sal = base_sal + 4.5 * (exp - mean_exp) + rng.normal(0.0, 3.0, size=n_per_group)
        remote = rng.choice(["Remote", "Hybrid", "On-Site"], size=n_per_group, p=[0.4, 0.4, 0.2])
        perf = np.clip(3.0 + 0.15 * (exp - mean_exp) + rng.normal(0.0, 0.4, size=n_per_group), 1.0, 5.0)

        rows_dept.extend([name] * n_per_group)
        rows_exp.extend(np.round(exp, 2).tolist())
        rows_sal.extend(np.round(sal, 2).tolist())
        rows_remote.extend(remote.tolist())
        rows_perf.extend(np.round(perf, 2).tolist())

    return pl.DataFrame(
        {
            "department": rows_dept,
            "years_experience": rows_exp,
            "salary_k": rows_sal,
            "work_mode": rows_remote,
            "performance_score": rows_perf,
        }
    )


def generate_saas_churn_dataset(n: int = 600, seed: int = 42) -> pl.DataFrame:
    """Generate a realistic SaaS Customer Churn dataset with known ground-truth effects:
    - `contract_type` ('Monthly' vs 'Annual') strongly impacts `churned` (Chi2 effect)
    - `support_tickets_30d` is significantly higher among churned customers (Skewed Mann-Whitney effect)
    - `monthly_spend_usd` varies across `plan_tier` ('Starter', 'Pro', 'Enterprise') (ANOVA effect)
    - `usage_hours_weekly` negatively correlates with `churn_risk_score` (Correlation effect)
    - `lucky_number_noise` is pure random noise (Should be REFUTED)
    """
    rng = np.random.default_rng(seed)

    plan_tier = rng.choice(["Starter", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15])
    contract_type = rng.choice(["Monthly", "Annual"], size=n, p=[0.6, 0.4])

    base_spend = np.where(plan_tier == "Enterprise", 450.0, np.where(plan_tier == "Pro", 140.0, 45.0))
    monthly_spend = np.round(np.clip(base_spend + rng.normal(0, 18.0, size=n), 15.0, 900.0), 2)

    usage_hours = np.round(np.clip(rng.gamma(shape=3.0, scale=4.5, size=n), 0.5, 60.0), 2)

    # Churn probability driven by Monthly contract + low usage
    logit_churn = (
        -1.2
        + 1.6 * (contract_type == "Monthly").astype(float)
        - 0.12 * usage_hours
        + rng.normal(0, 0.4, size=n)
    )
    prob_churn = 1.0 / (1.0 + np.exp(-logit_churn))
    churned_bool = rng.uniform(0, 1, size=n) < prob_churn
    churned = np.where(churned_bool, "Yes", "No")

    support_tickets = np.where(
        churned_bool,
        rng.poisson(lam=4.8, size=n),
        rng.poisson(lam=1.2, size=n),
    )

    churn_risk_score = np.round(np.clip(prob_churn * 100.0 + rng.normal(0, 5.0, size=n), 1.0, 99.0), 1)
    lucky_number_noise = np.round(rng.normal(50.0, 15.0, size=n), 2)

    return pl.DataFrame(
        {
            "plan_tier": plan_tier.tolist(),
            "contract_type": contract_type.tolist(),
            "monthly_spend_usd": monthly_spend.tolist(),
            "usage_hours_weekly": usage_hours.tolist(),
            "support_tickets_30d": support_tickets.tolist(),
            "churn_risk_score": churn_risk_score.tolist(),
            "churned": churned.tolist(),
            "lucky_number_noise": lucky_number_noise.tolist(),
        }
    )


def generate_nonlinear_and_confounded_dataset(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Generate a dataset with:
    1. A strong quadratic (inverted-U) relationship between `stress_index` and `productivity_score`
       (Pearson r ~ 0, but Quadratic OLS R² > 0.70!).
    2. A spurious correlation between `ice_cream_sales` and `sunburn_incidents` driven entirely
       by the confounder `ambient_temp_c`.
    """
    rng = np.random.default_rng(seed)

    # 1. Yerkes-Dodson inverted-U law: productivity peaks at moderate stress (stress=5)
    stress_index = rng.uniform(0.0, 10.0, size=n)
    productivity_score = 85.0 - 2.8 * ((stress_index - 5.0) ** 2) + rng.normal(0.0, 3.2, size=n)

    # 2. Confounder: ambient_temp_c drives both ice_cream_sales and sunburn_incidents
    ambient_temp_c = rng.normal(24.0, 6.0, size=n)
    ice_cream_sales = 120.0 + 15.0 * ambient_temp_c + rng.normal(0.0, 20.0, size=n)
    sunburn_incidents = np.clip(5.0 + 2.2 * ambient_temp_c + rng.normal(0.0, 4.0, size=n), 0.0, None)

    shift_group = np.where(stress_index > 5.0, "High_Load", "Balanced_Load")

    return pl.DataFrame(
        {
            "stress_index": np.round(stress_index, 3).tolist(),
            "productivity_score": np.round(productivity_score, 3).tolist(),
            "ambient_temp_c": np.round(ambient_temp_c, 2).tolist(),
            "ice_cream_sales": np.round(ice_cream_sales, 2).tolist(),
            "sunburn_incidents": np.round(sunburn_incidents, 2).tolist(),
            "shift_group": shift_group.tolist(),
        }
    )


BENCHMARK_DATASETS = {
    "SaaS Customer Churn (Multi-Factor)": (
        generate_saas_churn_dataset,
        "Identify the statistically significant drivers of customer churn, spending tiers, and support ticket volume while filtering out random noise.",
    ),
    "Simpson's Paradox (Salary vs Experience)": (
        generate_simpsons_paradox_dataset,
        "Investigate whether years of experience positively or negatively impacts salary, controlling for department confounding.",
    ),
    "Non-Linear U-Curve & Spurious Confounding": (
        generate_nonlinear_and_confounded_dataset,
        "Discover non-linear quadratic relationships (stress vs productivity) and test whether correlations survive confounding controls (temperature).",
    ),
}
