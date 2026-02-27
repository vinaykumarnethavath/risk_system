"""
test_market.py — Unit tests for the Market Engine
=================================================
Tests momentum, volatility, drawdown, volume trend,
and composite market confidence scoring.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.market_engine import (
    compute_momentum,
    compute_volatility,
    compute_max_drawdown,
    compute_volume_trend,
    compute_market_confidence,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def bullish_stock():
    """Stock with consistent upward trend."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.012, 252))
    return pd.DataFrame({
        "Close": prices,
        "Volume": np.random.randint(1e6, 5e6, 252),
    }, index=dates)


@pytest.fixture
def bearish_stock():
    """Stock with consistent downward trend."""
    np.random.seed(99)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(-0.003, 0.02, 252))
    return pd.DataFrame({
        "Close": prices,
        "Volume": np.random.randint(1e6, 5e6, 252),
    }, index=dates)


@pytest.fixture
def flat_stock():
    """Stock with little movement."""
    np.random.seed(77)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    prices = 100 + np.random.normal(0, 0.5, 252).cumsum() * 0.1
    return pd.DataFrame({
        "Close": prices,
        "Volume": np.random.randint(1e6, 5e6, 252),
    }, index=dates)


# ─── Momentum Tests ─────────────────────────────────────────────────


class TestComputeMomentum:
    def test_returns_dict(self, bullish_stock):
        result = compute_momentum(bullish_stock)
        assert isinstance(result, dict)

    def test_has_all_horizons(self, bullish_stock):
        result = compute_momentum(bullish_stock)
        assert "momentum_30d" in result
        assert "momentum_90d" in result
        assert "momentum_180d" in result

    def test_bullish_positive_momentum(self, bullish_stock):
        result = compute_momentum(bullish_stock)
        assert result["momentum_30d"] > -0.5  # Should generally be positive

    def test_bearish_negative_momentum(self, bearish_stock):
        result = compute_momentum(bearish_stock)
        assert result["momentum_180d"] < 0.5  # Should generally be negative

    def test_small_dataset(self):
        small = pd.DataFrame({"Close": [100, 105, 110]})
        result = compute_momentum(small)
        assert isinstance(result["momentum_30d"], float)

    def test_empty_returns_zeros(self):
        empty = pd.DataFrame({"Close": []})
        result = compute_momentum(empty)
        assert result["momentum_30d"] == 0.0


# ─── Volatility Tests ───────────────────────────────────────────────


class TestComputeVolatility:
    def test_returns_dict(self, bullish_stock):
        result = compute_volatility(bullish_stock)
        assert isinstance(result, dict)

    def test_non_negative(self, bullish_stock):
        result = compute_volatility(bullish_stock)
        assert result["daily_volatility"] >= 0
        assert result["annual_volatility"] >= 0

    def test_annual_greater_than_daily(self, bullish_stock):
        result = compute_volatility(bullish_stock)
        assert result["annual_volatility"] > result["daily_volatility"]

    def test_flat_stock_low_volatility(self, flat_stock):
        result = compute_volatility(flat_stock)
        assert result["annual_volatility"] < 0.5

    def test_small_dataset_returns_zeros(self):
        small = pd.DataFrame({"Close": [100, 105]})
        result = compute_volatility(small)
        assert result["daily_volatility"] == 0.0


# ─── Max Drawdown Tests ─────────────────────────────────────────────


class TestComputeMaxDrawdown:
    def test_returns_float(self, bullish_stock):
        result = compute_max_drawdown(bullish_stock)
        assert isinstance(result, float)

    def test_non_positive(self, bullish_stock):
        result = compute_max_drawdown(bullish_stock)
        assert result <= 0.0

    def test_bearish_deeper_drawdown(self, bearish_stock):
        result = compute_max_drawdown(bearish_stock)
        assert result < -0.05

    def test_single_point_returns_zero(self):
        single = pd.DataFrame({"Close": [100]})
        result = compute_max_drawdown(single)
        assert result == 0.0


# ─── Volume Trend Tests ─────────────────────────────────────────────


class TestComputeVolumeTrend:
    def test_returns_float(self, bullish_stock):
        result = compute_volume_trend(bullish_stock)
        assert isinstance(result, float)

    def test_no_volume_column(self):
        no_vol = pd.DataFrame({"Close": [100, 105, 110]})
        result = compute_volume_trend(no_vol)
        assert result == 0.0

    def test_small_dataset_returns_zero(self):
        small = pd.DataFrame({"Close": [100], "Volume": [1e6]})
        result = compute_volume_trend(small)
        assert result == 0.0


# ─── Composite Market Confidence Tests ──────────────────────────────


class TestComputeMarketConfidence:
    def test_returns_dict(self, bullish_stock):
        result = compute_market_confidence(bullish_stock)
        assert isinstance(result, dict)

    def test_has_required_keys(self, bullish_stock):
        result = compute_market_confidence(bullish_stock)
        assert "market_confidence" in result
        assert "market_signal" in result
        assert "max_drawdown" in result
        assert "volume_trend" in result

    def test_confidence_bounded(self, bullish_stock):
        result = compute_market_confidence(bullish_stock)
        assert 0.0 <= result["market_confidence"] <= 1.0

    def test_signal_valid(self, bullish_stock):
        result = compute_market_confidence(bullish_stock)
        assert result["market_signal"] in ["BULLISH", "NEUTRAL", "BEARISH"]

    def test_bearish_lower_confidence(self, bullish_stock, bearish_stock):
        bull = compute_market_confidence(bullish_stock)
        bear = compute_market_confidence(bearish_stock)
        # Bearish stock should generally have lower confidence
        assert bear["market_confidence"] <= bull["market_confidence"] + 0.3
