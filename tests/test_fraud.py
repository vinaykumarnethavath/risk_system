"""
test_fraud.py — Unit tests for the Fraud Engine
================================================
Tests Benford's Law deviation, statistical anomaly scoring,
Isolation Forest anomaly detection, and composite fraud probability.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.fraud_engine import (
    benford_deviation,
    statistical_anomaly_score,
    isolation_forest_score,
    compute_fraud_probability,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def clean_financials():
    """Realistic, non-fraudulent quarterly financials."""
    return pd.DataFrame({
        "Total Revenue": [1e9, 1.05e9, 1.1e9, 1.15e9, 1.2e9, 1.25e9, 1.3e9, 1.35e9],
        "Net Income": [1e8, 1.05e8, 1.1e8, 1.12e8, 1.18e8, 1.2e8, 1.25e8, 1.3e8],
        "Operating Expense": [8e8, 8.2e8, 8.5e8, 8.7e8, 8.9e8, 9.1e8, 9.3e8, 9.5e8],
    })


@pytest.fixture
def suspicious_financials():
    """Financials with anomalous patterns (large jumps, inconsistencies)."""
    return pd.DataFrame({
        "Total Revenue": [1e9, 2e9, 5e8, 3e9, 1e9, 4e9, 2e9, 6e9],
        "Net Income": [1e8, 5e8, -2e8, 8e8, -1e8, 1e9, -5e8, 2e9],
        "Operating Expense": [8e8, 1.2e9, 5e8, 2e9, 8e8, 2.5e9, 1.5e9, 3e9],
    })


# ─── Benford's Law Tests ────────────────────────────────────────────


class TestBenfordDeviation:
    def test_returns_float(self, clean_financials):
        result = benford_deviation(clean_financials["Total Revenue"])
        assert isinstance(result, float)

    def test_non_negative(self, clean_financials):
        result = benford_deviation(clean_financials["Total Revenue"])
        assert result >= 0.0

    def test_empty_series_returns_zero(self):
        result = benford_deviation(pd.Series([], dtype=float))
        assert result == 0.0

    def test_too_few_values_returns_zero(self):
        result = benford_deviation(pd.Series([100, 200, 300]))
        assert result == 0.0

    def test_handles_negative_values(self):
        series = pd.Series([-100, -200, 0, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        result = benford_deviation(series)
        assert isinstance(result, float)

    def test_handles_nan_values(self):
        series = pd.Series([np.nan, 100, 200, np.nan, 400, 500, 600, 700, 800, 900, 1000, 1100])
        result = benford_deviation(series)
        assert isinstance(result, float)


# ─── Statistical Anomaly Tests ──────────────────────────────────────


class TestStatisticalAnomalyScore:
    def test_returns_dict(self, clean_financials):
        result = statistical_anomaly_score(clean_financials)
        assert isinstance(result, dict)

    def test_has_required_keys(self, clean_financials):
        result = statistical_anomaly_score(clean_financials)
        assert "revenue_volatility" in result
        assert "margin_stability" in result
        assert "jump_score" in result
        assert "statistical_risk" in result

    def test_statistical_risk_bounded(self, clean_financials):
        result = statistical_anomaly_score(clean_financials)
        assert 0.0 <= result["statistical_risk"] <= 1.0

    def test_clean_data_low_volatility(self, clean_financials):
        result = statistical_anomaly_score(clean_financials)
        assert result["revenue_volatility"] < 0.2

    def test_suspicious_data_higher_risk(self, suspicious_financials):
        result = statistical_anomaly_score(suspicious_financials)
        assert result["jump_score"] > 0.0

    def test_empty_dataframe(self):
        empty = pd.DataFrame({"Total Revenue": []})
        result = statistical_anomaly_score(empty)
        assert result["statistical_risk"] == 0.0


# ─── Isolation Forest Tests ─────────────────────────────────────────


class TestIsolationForestScore:
    def test_returns_float(self, clean_financials):
        result = isolation_forest_score(clean_financials)
        assert isinstance(result, float)

    def test_bounded_zero_one(self, clean_financials):
        result = isolation_forest_score(clean_financials)
        assert 0.0 <= result <= 1.0

    def test_empty_returns_zero(self):
        empty = pd.DataFrame()
        result = isolation_forest_score(empty)
        assert result == 0.0

    def test_few_rows_returns_zero(self):
        small = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = isolation_forest_score(small)
        assert result == 0.0


# ─── Composite Fraud Probability Tests ──────────────────────────────


class TestComputeFraudProbability:
    def test_returns_dict(self, clean_financials):
        result = compute_fraud_probability(clean_financials)
        assert isinstance(result, dict)

    def test_has_required_keys(self, clean_financials):
        result = compute_fraud_probability(clean_financials)
        assert "fraud_probability" in result
        assert "benford_score" in result
        assert "benford_flag" in result
        assert "statistical_scores" in result
        assert "isolation_score" in result
        assert "risk_level" in result

    def test_probability_bounded(self, clean_financials):
        result = compute_fraud_probability(clean_financials)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_risk_level_valid(self, clean_financials):
        result = compute_fraud_probability(clean_financials)
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_suspicious_data_higher_fraud(self, clean_financials, suspicious_financials):
        clean_result = compute_fraud_probability(clean_financials)
        suspicious_result = compute_fraud_probability(suspicious_financials)
        # Suspicious data should generally score higher
        assert suspicious_result["statistical_scores"]["statistical_risk"] >= clean_result["statistical_scores"]["statistical_risk"]

    def test_benford_flag_consistency(self, clean_financials):
        result = compute_fraud_probability(clean_financials)
        if result["benford_score"] > 0.25:
            assert result["benford_flag"] is True
        else:
            assert result["benford_flag"] is False
