"""
market_engine.py — Market Confidence Scoring Engine
====================================================
Computes market-based confidence signals from stock price data:
  - Momentum (short and long-term trends)
  - Volatility (risk measurement)
  - Maximum drawdown (worst-case loss)
  - Volume trends (institutional interest)

These signals combine into a single market confidence score.
"""

import numpy as np
import pandas as pd


# ─── Individual Signals ─────────────────────────────────────────────


def compute_momentum(stock_df: pd.DataFrame) -> dict:
    """
    Compute price momentum over multiple time horizons.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Stock OHLCV data with 'Close' column.

    Returns
    -------
    dict
        Momentum values for 30-day, 90-day, and 180-day windows.
    """
    close = stock_df["Close"].dropna()
    if len(close) < 5:
        return {"momentum_30d": 0.0, "momentum_90d": 0.0, "momentum_180d": 0.0}

    result = {}
    for label, days in [("30d", 30), ("90d", 90), ("180d", 180)]:
        if len(close) >= days:
            mom = float(close.iloc[-1]) / float(close.iloc[-days]) - 1
        else:
            mom = float(close.iloc[-1]) / float(close.iloc[0]) - 1
        result[f"momentum_{label}"] = round(mom, 4)

    return result


def compute_volatility(stock_df: pd.DataFrame) -> dict:
    """
    Compute price volatility metrics.

    Returns
    -------
    dict
        Daily volatility, annualised volatility, and
        volatility percentile rank.
    """
    close = stock_df["Close"].dropna()
    if len(close) < 10:
        return {"daily_volatility": 0.0, "annual_volatility": 0.0}

    returns = close.pct_change().dropna()
    daily_vol = float(returns.std().iloc[0]) if hasattr(returns.std(), 'iloc') else float(returns.std())
    annual_vol = daily_vol * np.sqrt(252)

    return {
        "daily_volatility": round(daily_vol, 6),
        "annual_volatility": round(annual_vol, 4),
    }


def compute_max_drawdown(stock_df: pd.DataFrame) -> float:
    """
    Compute the maximum drawdown (worst peak-to-trough decline).

    Returns
    -------
    float
        Maximum drawdown as a negative fraction (e.g. -0.25 = -25%).
    """
    close = stock_df["Close"].dropna()
    if len(close) < 2:
        return 0.0

    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    drawdown_min = drawdown.min()
    return round(float(drawdown_min.iloc[0]) if hasattr(drawdown_min, 'iloc') else float(drawdown_min), 4)


def compute_volume_trend(stock_df: pd.DataFrame) -> float:
    """
    Compute recent volume trend relative to historical average.

    Positive values indicate increasing interest; negative values
    indicate declining interest.

    Returns
    -------
    float
        Volume trend ratio (recent_avg / historical_avg - 1).
    """
    if "Volume" not in stock_df.columns:
        return 0.0

    vol = stock_df["Volume"].dropna()
    if len(vol) < 30:
        return 0.0

    recent_avg = vol.tail(20).mean()
    historical_avg = vol.mean()

    if float(historical_avg) == 0:
        return 0.0

    return round(float(recent_avg) / float(historical_avg) - 1, 4)


# ─── Composite Market Confidence ────────────────────────────────────


def compute_market_confidence(stock_df: pd.DataFrame) -> dict:
    """
    Compute a composite market confidence score from all signals.

    Combines:
    - 30-day momentum (positive = bullish)
    - Volatility penalty (high vol = less confidence)
    - Drawdown penalty (deep drawdown = risk)
    - Volume trend bonus (rising volume = institutional interest)

    Parameters
    ----------
    stock_df : pd.DataFrame
        Stock OHLCV data.

    Returns
    -------
    dict
        Keys: 'market_confidence' (0–1 normalised score),
              plus all individual signal values.
    """
    momentum = compute_momentum(stock_df)
    volatility = compute_volatility(stock_df)
    max_dd = compute_max_drawdown(stock_df)
    vol_trend = compute_volume_trend(stock_df)

    # Raw confidence = momentum - volatility penalty - drawdown penalty + volume bonus
    raw = (
        momentum["momentum_30d"]
        - volatility["annual_volatility"] * 0.5
        - abs(max_dd) * 0.3
        + vol_trend * 0.1
    )

    # Normalise to 0–1 using sigmoid-like transformation
    confidence = 1 / (1 + np.exp(-5 * raw))
    confidence = round(float(confidence), 4)

    # Risk level
    if confidence >= 0.65:
        level = "BULLISH"
    elif confidence >= 0.45:
        level = "NEUTRAL"
    else:
        level = "BEARISH"

    return {
        "market_confidence": confidence,
        "market_signal": level,
        **momentum,
        **volatility,
        "max_drawdown": max_dd,
        "volume_trend": vol_trend,
    }


if __name__ == "__main__":
    import json
    import yfinance as yf

    stock = yf.download("AAPL", period="1y", progress=False)
    result = compute_market_confidence(stock)
    print(json.dumps(result, indent=2, default=str))
