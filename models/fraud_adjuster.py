"""
fraud_adjuster.py — Fraud-Adjusted Financial Correction Layer
=============================================================
Applies trust-weighted corrections to raw financials based on
the fraud probability computed by the fraud engine.

This is the core innovation layer: rather than discarding
suspicious data, we discount it proportionally to the detected
fraud risk, producing "fraud-adjusted" financials.
"""

import numpy as np
import pandas as pd


# ─── Core Adjustment ────────────────────────────────────────────────


def adjust_financials(
    financials: pd.DataFrame, fraud_prob: float
) -> pd.DataFrame:
    """
    Apply a trust-weighted adjustment to raw financial figures.

    Trust = 1 − fraud_probability. All numeric financial values are
    scaled by the trust factor, effectively discounting figures that
    come from high-fraud-risk companies.

    Parameters
    ----------
    financials : pd.DataFrame
        Raw quarterly financial statements.
    fraud_prob : float
        Fraud probability from the fraud engine (0–1).

    Returns
    -------
    pd.DataFrame
        Fraud-adjusted financial statements.
    """
    trust = 1.0 - fraud_prob
    adjusted = financials.copy()
    numeric_cols = adjusted.select_dtypes(include=[np.number]).columns
    adjusted[numeric_cols] = adjusted[numeric_cols] * trust
    return adjusted


# ─── Growth Metrics ─────────────────────────────────────────────────


def compute_adjusted_growth(
    financials: pd.DataFrame, fraud_prob: float
) -> dict:
    """
    Compute fraud-adjusted growth metrics.

    Parameters
    ----------
    financials : pd.DataFrame
        Raw quarterly financial statements.
    fraud_prob : float
        Fraud probability (0–1).

    Returns
    -------
    dict
        Keys: 'raw_growth', 'adjusted_growth', 'trust_factor',
              'fraud_discount', 'adjustment_pct'.
    """
    trust = 1.0 - fraud_prob
    result = {"trust_factor": round(trust, 4), "fraud_probability": round(fraud_prob, 4)}

    # Revenue growth
    rev_col = None
    for col in ["Total Revenue", "Revenue", "total_revenue"]:
        if col in financials.columns:
            rev_col = col
            break

    if rev_col is not None:
        # Sort index ascending so iloc[0] is oldest, iloc[-1] is newest
        rev = financials[rev_col].dropna().sort_index()
        if len(rev) >= 2:
            raw_growth = float((rev.iloc[-1] - rev.iloc[0]) / abs(rev.iloc[0] + 1e-10))
            if raw_growth >= 0:
                adjusted_growth = raw_growth * trust
            else:
                adjusted_growth = raw_growth * (2.0 - trust)
            result["raw_growth"] = round(raw_growth, 4)
            result["adjusted_growth"] = round(adjusted_growth, 4)
            result["fraud_discount"] = round(raw_growth - adjusted_growth, 4)
            result["adjustment_pct"] = round((1 - trust) * 100, 2)
        else:
            result["raw_growth"] = 0.0
            result["adjusted_growth"] = 0.0
            result["fraud_discount"] = 0.0
            result["adjustment_pct"] = 0.0
    else:
        result["raw_growth"] = 0.0
        result["adjusted_growth"] = 0.0
        result["fraud_discount"] = 0.0
        result["adjustment_pct"] = 0.0

    return result


# ─── Detailed Adjustment Report ─────────────────────────────────────


def generate_adjustment_report(
    financials: pd.DataFrame, fraud_prob: float
) -> dict:
    """
    Generate a comprehensive fraud adjustment report.

    Parameters
    ----------
    financials : pd.DataFrame
        Raw quarterly financial statements.
    fraud_prob : float
        Fraud probability (0–1).

    Returns
    -------
    dict
        Full report with raw vs adjusted figures, growth metrics,
        and risk commentary.
    """
    adjusted = adjust_financials(financials, fraud_prob)
    growth = compute_adjusted_growth(financials, fraud_prob)

    report = {
        "growth_metrics": growth,
        "quarters_analysed": len(financials),
        "columns_adjusted": list(
            financials.select_dtypes(include=[np.number]).columns
        ),
    }

    # Summary statistics: raw vs adjusted
    numeric_cols = financials.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        raw_total = float(financials[numeric_cols].sum().sum())
        adj_total = float(adjusted[numeric_cols].sum().sum())
        report["raw_total"] = round(raw_total, 2)
        report["adjusted_total"] = round(adj_total, 2)
        report["total_discount"] = round(raw_total - adj_total, 2)

    # Risk commentary
    if fraud_prob >= 0.6:
        report["commentary"] = (
            "HIGH FRAUD RISK: Financial figures discounted significantly. "
            "Recommend manual forensic audit before investment decisions."
        )
    elif fraud_prob >= 0.3:
        report["commentary"] = (
            "MODERATE FRAUD RISK: Financial figures partially discounted. "
            "Some anomalies detected — exercise caution."
        )
    else:
        report["commentary"] = (
            "LOW FRAUD RISK: Financial figures largely trustworthy. "
            "Minor adjustments applied as standard precaution."
        )

    return report


if __name__ == "__main__":
    import json

    sample = pd.DataFrame({
        "Total Revenue": [1e9, 1.1e9, 1.15e9, 1.2e9],
        "Net Income": [1e8, 1.1e8, 1.05e8, 1.2e8],
        "Operating Expense": [8e8, 8.5e8, 9e8, 8.8e8],
    })

    report = generate_adjustment_report(sample, fraud_prob=0.35)
    print(json.dumps(report, indent=2, default=str))
