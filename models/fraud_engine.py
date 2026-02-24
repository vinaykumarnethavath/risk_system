"""
fraud_engine.py — Unified Fraud Probability Engine
===================================================
Connects forensic analytics from the Hybrid Risk module into a
clean interface for computing fraud probabilities on financial data.

Implements three forensic signal layers:
  1. Benford's Law deviation scoring
  2. Statistical anomaly detection (z-scores, volatility)
  3. Isolation Forest unsupervised anomaly scoring

These signals are combined into a single fraud probability estimate.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import ISOLATION_FOREST_CONTAMINATION, BENFORD_THRESHOLD


# ─── Forensic Signal 1: Benford's Law ───────────────────────────────


def benford_deviation(values: pd.Series) -> float:
    """
    Compute Benford's Law deviation for a series of financial figures.

    Natural financial data follows a predictable first-digit distribution.
    Manipulated data deviates significantly from this pattern.

    Parameters
    ----------
    values : pd.Series
        Financial figures (revenue, expenses, etc.).

    Returns
    -------
    float
        L2-norm deviation from expected Benford distribution.
        Higher values indicate higher manipulation risk.
    """
    def first_digit(n):
        try:
            n = abs(float(n))
            if n == 0 or pd.isna(n):
                return np.nan
            return int(str(n).lstrip("0").lstrip(".")[0])
        except (ValueError, IndexError):
            return np.nan

    digits = values.apply(first_digit).dropna()
    if len(digits) < 10:
        return 0.0

    observed = digits.value_counts(normalize=True).reindex(
        range(1, 10), fill_value=0
    )
    expected = np.log10(1 + 1 / np.arange(1, 10))
    deviation = float(np.linalg.norm(observed.values - expected))
    return deviation


# ─── Forensic Signal 2: Statistical Anomaly Scoring ─────────────────


def statistical_anomaly_score(financials: pd.DataFrame) -> dict:
    """
    Detect statistical anomalies in financial time series.

    Checks for:
    - Revenue volatility spikes
    - Profit margin inconsistencies
    - Unusual quarter-over-quarter jumps

    Parameters
    ----------
    financials : pd.DataFrame
        Quarterly financial statements (rows = quarters).

    Returns
    -------
    dict
        Keys: 'revenue_volatility', 'margin_stability',
              'jump_score', 'statistical_risk'.
    """
    scores = {}

    # Revenue volatility
    if "Total Revenue" in financials.columns:
        rev = financials["Total Revenue"].dropna()
        if len(rev) >= 4:
            pct_changes = rev.pct_change().dropna()
            scores["revenue_volatility"] = float(pct_changes.std())

            # Large quarter-over-quarter jumps (>50% change)
            jumps = (pct_changes.abs() > 0.5).sum()
            scores["jump_score"] = float(jumps / len(pct_changes))
        else:
            scores["revenue_volatility"] = 0.0
            scores["jump_score"] = 0.0
    else:
        scores["revenue_volatility"] = 0.0
        scores["jump_score"] = 0.0

    # Profit margin stability
    if (
        "Net Income" in financials.columns
        and "Total Revenue" in financials.columns
    ):
        rev = financials["Total Revenue"].replace(0, np.nan)
        margin = financials["Net Income"] / rev
        margin = margin.dropna()
        if len(margin) >= 4:
            scores["margin_stability"] = float(margin.std())
        else:
            scores["margin_stability"] = 0.0
    else:
        scores["margin_stability"] = 0.0

    # Composite statistical risk (0 to 1)
    risk = min(
        1.0,
        (
            scores["revenue_volatility"] * 0.4
            + scores["margin_stability"] * 0.3
            + scores["jump_score"] * 0.3
        ),
    )
    scores["statistical_risk"] = round(risk, 4)

    return scores


# ─── Forensic Signal 3: Isolation Forest Anomaly ────────────────────


def isolation_forest_score(financials: pd.DataFrame) -> float:
    """
    Use Isolation Forest to detect anomalous financial patterns.

    Parameters
    ----------
    financials : pd.DataFrame
        Quarterly financial data.

    Returns
    -------
    float
        Anomaly score between 0 (normal) and 1 (highly anomalous).
    """
    # Select numeric columns only
    numeric = financials.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.empty or len(numeric) < 4:
        return 0.0

    # Fill remaining NaNs with column medians
    numeric = numeric.fillna(numeric.median())

    iso = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    try:
        # decision_function returns negative scores for anomalies
        raw_scores = iso.fit(numeric).decision_function(numeric)
        # Convert to 0-1 scale (lower decision_function → higher anomaly)
        normalised = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )
        # Return the mean anomaly across all quarters
        return float(np.mean(normalised))
    except Exception:
        return 0.0


# ─── Unified Fraud Probability ──────────────────────────────────────


def compute_fraud_probability(financials: pd.DataFrame) -> dict:
    """
    Compute a unified fraud probability by combining three
    forensic signal layers.

    Parameters
    ----------
    financials : pd.DataFrame
        Quarterly financial statements.

    Returns
    -------
    dict
        Keys:
        - 'fraud_probability': float (0–1), composite score
        - 'benford_score': float, Benford deviation
        - 'benford_flag': bool, True if above threshold
        - 'statistical_scores': dict, detailed statistical anomaly scores
        - 'isolation_score': float, Isolation Forest anomaly score
        - 'risk_level': str, 'LOW' / 'MEDIUM' / 'HIGH'
    """
    result = {}

    # Signal 1: Benford's Law
    revenue_col = None
    for col in ["Total Revenue", "Revenue", "total_revenue"]:
        if col in financials.columns:
            revenue_col = col
            break

    if revenue_col is not None:
        benford = benford_deviation(financials[revenue_col])
    else:
        # Try first numeric column as fallback
        num_cols = financials.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            benford = benford_deviation(financials[num_cols[0]])
        else:
            benford = 0.0

    result["benford_score"] = round(benford, 4)
    result["benford_flag"] = benford > BENFORD_THRESHOLD

    # Signal 2: Statistical anomalies
    stats = statistical_anomaly_score(financials)
    result["statistical_scores"] = stats

    # Signal 3: Isolation Forest
    iso_score = isolation_forest_score(financials)
    result["isolation_score"] = round(iso_score, 4)

    # Composite fraud probability (weighted combination)
    benford_norm = min(1.0, benford / 0.5)  # Normalise to 0–1
    fraud_prob = (
        0.35 * benford_norm
        + 0.35 * stats["statistical_risk"]
        + 0.30 * iso_score
    )
    fraud_prob = round(min(1.0, max(0.0, fraud_prob)), 4)

    result["fraud_probability"] = fraud_prob

    # Risk level classification
    if fraud_prob >= 0.6:
        result["risk_level"] = "HIGH"
    elif fraud_prob >= 0.3:
        result["risk_level"] = "MEDIUM"
    else:
        result["risk_level"] = "LOW"

    return result


if __name__ == "__main__":
    # Quick test with sample data
    import json

    sample = pd.DataFrame({
        "Total Revenue": [1e9, 1.1e9, 1.15e9, 1.2e9, 1.3e9, 1.25e9, 1.4e9, 1.5e9],
        "Net Income": [1e8, 1.1e8, 1.05e8, 1.2e8, 1.3e8, 1.1e8, 1.4e8, 1.5e8],
        "Operating Expense": [8e8, 8.5e8, 9e8, 8.8e8, 9.2e8, 9.5e8, 9.8e8, 1e9],
    })
    result = compute_fraud_probability(sample)
    print(json.dumps(result, indent=2, default=str))
