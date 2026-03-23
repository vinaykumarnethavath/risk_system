"""
feature_engineering.py — Optimized Feature Engineering Module
=============================================================
Advanced feature extraction for improved fraud detection accuracy.
Provides reusable feature engineering functions that can be applied
to financial data from any source.

Enhancements over base fraud_engine:
  - Altman Z-Score for bankruptcy prediction
  - Beneish M-Score for earnings manipulation detection
  - Cash flow quality metrics
  - Revenue-expense divergence analysis
  - Rolling window forensic features
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ─── Altman Z-Score ──────────────────────────────────────────────────


def compute_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    total_assets: float,
    revenue: float,
) -> dict:
    """
    Compute the Altman Z-Score for bankruptcy prediction.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Cap / Total Liabilities
        X5 = Revenue / Total Assets

    Interpretation:
        Z > 2.99 → Safe zone
        1.81 < Z < 2.99 → Grey zone
        Z < 1.81 → Distress zone

    Returns
    -------
    dict
        z_score, zone, components.
    """
    if total_assets == 0:
        return {"z_score": 0.0, "zone": "UNKNOWN", "components": {}}

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / (total_liabilities + 1e-10)
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "SAFE"
    elif z > 1.81:
        zone = "GREY"
    else:
        zone = "DISTRESS"

    return {
        "z_score": round(float(z), 4),
        "zone": zone,
        "components": {
            "X1_working_capital_ratio": round(x1, 4),
            "X2_retained_earnings_ratio": round(x2, 4),
            "X3_ebit_ratio": round(x3, 4),
            "X4_market_leverage": round(x4, 4),
            "X5_asset_turnover": round(x5, 4),
        },
    }


# ─── Beneish M-Score ────────────────────────────────────────────────


def compute_beneish_m_score(
    current: dict, prior: dict
) -> dict:
    """
    Compute the Beneish M-Score for earnings manipulation detection.

    M = −4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI − 0.172*SGAI + 4.679*TATA − 0.327*LVGI

    M > −1.78 suggests high probability of manipulation.

    Parameters
    ----------
    current : dict
        Current period financials with keys:
        revenue, cogs, receivables, current_assets, ppe,
        depreciation, sga, total_assets, long_term_debt,
        current_liabilities, net_income, cfo.
    prior : dict
        Prior period financials with same keys.

    Returns
    -------
    dict
        m_score, manipulation_likely, components.
    """
    def safe_div(a, b):
        return a / b if b != 0 else 0

    # Days Sales in Receivables Index
    dsri = safe_div(
        safe_div(current.get("receivables", 0), current.get("revenue", 1)),
        safe_div(prior.get("receivables", 0), prior.get("revenue", 1))
    )

    # Gross Margin Index
    curr_gm = 1 - safe_div(current.get("cogs", 0), current.get("revenue", 1))
    prior_gm = 1 - safe_div(prior.get("cogs", 0), prior.get("revenue", 1))
    gmi = safe_div(prior_gm, curr_gm)

    # Asset Quality Index
    curr_aq = 1 - safe_div(
        current.get("current_assets", 0) + current.get("ppe", 0),
        current.get("total_assets", 1)
    )
    prior_aq = 1 - safe_div(
        prior.get("current_assets", 0) + prior.get("ppe", 0),
        prior.get("total_assets", 1)
    )
    aqi = safe_div(curr_aq, prior_aq)

    # Sales Growth Index
    sgi = safe_div(current.get("revenue", 0), prior.get("revenue", 1))

    # Depreciation Index
    curr_dep = safe_div(
        current.get("depreciation", 0),
        current.get("depreciation", 0) + current.get("ppe", 1)
    )
    prior_dep = safe_div(
        prior.get("depreciation", 0),
        prior.get("depreciation", 0) + prior.get("ppe", 1)
    )
    depi = safe_div(prior_dep, curr_dep)

    # SGA Expense Index
    curr_sgai = safe_div(current.get("sga", 0), current.get("revenue", 1))
    prior_sgai = safe_div(prior.get("sga", 0), prior.get("revenue", 1))
    sgai = safe_div(curr_sgai, prior_sgai)

    # Total Accruals to Total Assets
    tata = safe_div(
        current.get("net_income", 0) - current.get("cfo", 0),
        current.get("total_assets", 1)
    )

    # Leverage Index
    curr_lev = safe_div(
        current.get("long_term_debt", 0) + current.get("current_liabilities", 0),
        current.get("total_assets", 1)
    )
    prior_lev = safe_div(
        prior.get("long_term_debt", 0) + prior.get("current_liabilities", 0),
        prior.get("total_assets", 1)
    )
    lvgi = safe_div(curr_lev, prior_lev)

    # Composite M-Score
    m = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    return {
        "m_score": round(float(m), 4),
        "manipulation_likely": m > -1.78,
        "components": {
            "DSRI": round(dsri, 4),
            "GMI": round(gmi, 4),
            "AQI": round(aqi, 4),
            "SGI": round(sgi, 4),
            "DEPI": round(depi, 4),
            "SGAI": round(sgai, 4),
            "TATA": round(tata, 4),
            "LVGI": round(lvgi, 4),
        },
    }


# ─── Cash Flow Quality ──────────────────────────────────────────────


def compute_cash_flow_quality(financials: pd.DataFrame) -> dict:
    """
    Assess cash flow quality — divergence between reported earnings
    and actual cash flows is a classic manipulation signal.

    Parameters
    ----------
    financials : pd.DataFrame
        Must contain 'Net Income' and ideally 'Operating Cash Flow'
        or 'Free Cash Flow'.

    Returns
    -------
    dict
        accrual_ratio, cash_conversion, quality_score.
    """
    result = {}

    net_income = financials.get("Net Income", pd.Series(dtype=float)).dropna()

    # Check for operating cash flow under common column names
    cfo_col = None
    for col in ["Operating Cash Flow", "Cash Flow From Operations", "Total Cash From Operating Activities"]:
        if col in financials.columns:
            cfo_col = col
            break

    if cfo_col and len(net_income) >= 2:
        cfo = financials[cfo_col].dropna()
        min_len = min(len(net_income), len(cfo))
        net_income = net_income.iloc[:min_len]
        cfo = cfo.iloc[:min_len]

        # Accrual ratio: (Net Income - CFO) / Total Assets proxy
        accruals = net_income.values - cfo.values
        avg_ni = np.mean(np.abs(net_income.values)) + 1e-10
        result["accrual_ratio"] = round(float(np.mean(accruals) / avg_ni), 4)

        # Cash conversion: CFO / Net Income (>1 is healthy)
        ni_safe = net_income.replace(0, np.nan).dropna()
        cfo_aligned = cfo.iloc[:len(ni_safe)]
        if len(ni_safe) > 0:
            conversion = float(np.mean(cfo_aligned.values / ni_safe.values))
            result["cash_conversion"] = round(conversion, 4)
        else:
            result["cash_conversion"] = 0.0
    else:
        result["accrual_ratio"] = 0.0
        result["cash_conversion"] = 1.0

    # Quality score (0 = suspicious, 1 = healthy)
    accrual_penalty = min(1.0, abs(result["accrual_ratio"]))
    conversion_score = min(1.0, max(0.0, result["cash_conversion"]))
    result["quality_score"] = round(
        0.5 * (1 - accrual_penalty) + 0.5 * conversion_score, 4
    )

    return result


# ─── Revenue-Expense Divergence ─────────────────────────────────────


def compute_revenue_expense_divergence(financials: pd.DataFrame) -> dict:
    """
    Detect unnatural divergence between revenue and expense trends.

    In healthy companies, revenue and expenses generally move together.
    Divergence (growing revenue with flat/declining costs) may indicate
    fabricated revenue or hidden expenses.

    Returns
    -------
    dict
        divergence_score, revenue_trend, expense_trend, correlation.
    """
    result = {}

    rev = financials.get("Total Revenue", pd.Series(dtype=float)).dropna()
    exp = financials.get("Operating Expense", pd.Series(dtype=float)).dropna()

    if len(rev) >= 4 and len(exp) >= 4:
        min_len = min(len(rev), len(exp))
        rev = rev.iloc[:min_len]
        exp = exp.iloc[:min_len]

        # Normalised trends
        rev_norm = (rev - rev.mean()) / (rev.std() + 1e-10)
        exp_norm = (exp - exp.mean()) / (exp.std() + 1e-10)

        # Correlation (should be high for healthy companies)
        correlation = float(rev_norm.corr(exp_norm))
        result["correlation"] = round(correlation, 4)

        # Growth rates
        rev_growth = float(rev.pct_change().mean())
        exp_growth = float(exp.pct_change().mean())
        result["revenue_trend"] = round(rev_growth, 4)
        result["expense_trend"] = round(exp_growth, 4)

        # Divergence: low correlation or mismatched growth = suspicious
        corr_penalty = max(0, 1 - abs(correlation))
        growth_mismatch = abs(rev_growth - exp_growth)
        result["divergence_score"] = round(
            0.6 * corr_penalty + 0.4 * min(1.0, growth_mismatch * 5), 4
        )
    else:
        result["correlation"] = 1.0
        result["revenue_trend"] = 0.0
        result["expense_trend"] = 0.0
        result["divergence_score"] = 0.0

    return result


# ─── Rolling Window Features ────────────────────────────────────────


def compute_rolling_forensic_features(
    financials: pd.DataFrame, window: int = 4
) -> pd.DataFrame:
    """
    Compute rolling window forensic features for time-aware detection.

    Parameters
    ----------
    financials : pd.DataFrame
        Quarterly financial data.
    window : int
        Rolling window size in quarters (default 4 = 1 year).

    Returns
    -------
    pd.DataFrame
        Additional forensic feature columns.
    """
    features = pd.DataFrame(index=financials.index)

    if "Total Revenue" in financials.columns:
        rev = financials["Total Revenue"]
        features["rev_rolling_mean"] = rev.rolling(window, min_periods=2).mean()
        features["rev_rolling_std"] = rev.rolling(window, min_periods=2).std()
        features["rev_zscore"] = (rev - features["rev_rolling_mean"]) / (
            features["rev_rolling_std"] + 1e-10
        )
        features["rev_acceleration"] = rev.pct_change().diff()

    if "Net Income" in financials.columns:
        ni = financials["Net Income"]
        features["ni_rolling_mean"] = ni.rolling(window, min_periods=2).mean()
        features["ni_volatility"] = ni.rolling(window, min_periods=2).std() / (
            ni.rolling(window, min_periods=2).mean().abs() + 1e-10
        )

    if (
        "Net Income" in financials.columns
        and "Total Revenue" in financials.columns
    ):
        margin = financials["Net Income"] / financials["Total Revenue"].replace(0, np.nan)
        features["margin_stability"] = margin.rolling(window, min_periods=2).std()
        features["margin_trend"] = margin.diff().rolling(window, min_periods=2).mean()

    return features.fillna(0)


# ─── Aggregated Feature Pipeline ────────────────────────────────────


def extract_enhanced_features(financials: pd.DataFrame) -> dict:
    """
    Run the full enhanced feature extraction pipeline.

    Returns
    -------
    dict
        All computed features: cash_flow_quality, divergence,
        rolling forensic features summary.
    """
    result = {}

    result["cash_flow_quality"] = compute_cash_flow_quality(financials)
    result["revenue_expense_divergence"] = compute_revenue_expense_divergence(financials)

    rolling = compute_rolling_forensic_features(financials)
    if not rolling.empty:
        result["rolling_features_summary"] = {
            col: {
                "mean": round(float(rolling[col].mean()), 4),
                "std": round(float(rolling[col].std()), 4),
                "max_abs": round(float(rolling[col].abs().max()), 4),
            }
            for col in rolling.columns
        }

    return result


if __name__ == "__main__":
    # Demo with sample data
    sample = pd.DataFrame({
        "Total Revenue": [1e9, 1.1e9, 1.15e9, 1.2e9, 1.3e9, 1.25e9, 1.4e9, 1.5e9],
        "Net Income": [1e8, 1.1e8, 1.05e8, 1.2e8, 1.3e8, 1.1e8, 1.4e8, 1.5e8],
        "Operating Expense": [8e8, 8.5e8, 9e8, 8.8e8, 9.2e8, 9.5e8, 9.8e8, 1e9],
    })

    features = extract_enhanced_features(sample)
    print(json.dumps(features, indent=2, default=str))

    # Altman Z-Score demo
    z = compute_altman_z_score(
        working_capital=5e8,
        retained_earnings=2e9,
        ebit=3e8,
        market_cap=1e10,
        total_liabilities=4e9,
        total_assets=8e9,
        revenue=6e9,
    )
    print(f"\nAltman Z-Score: {z['z_score']} ({z['zone']})")

    import json
