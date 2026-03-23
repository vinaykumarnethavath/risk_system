"""
peer_engine.py — Peer-Relative Performance Analysis Module
==========================================================
Compares a target company against its sector peers to determine
relative strength across financial and market dimensions.

Supports:
  - Single-metric peer comparison
  - Multi-dimensional peer scoring
  - Percentile ranking within peer group
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import DEFAULT_PEERS


# ─── Single Metric Comparison ───────────────────────────────────────


def compute_peer_strength(metric: float, peer_metrics: list) -> dict:
    """
    Compare a single metric against peer values.

    Parameters
    ----------
    metric : float
        Target company metric value.
    peer_metrics : list of float
        Same metric for peer companies.

    Returns
    -------
    dict
        Keys: 'value', 'peer_mean', 'peer_std', 'z_score',
              'percentile', 'relative_strength'.
    """
    peer_arr = np.array([p for p in peer_metrics if p is not None and not np.isnan(p)])

    if len(peer_arr) == 0:
        return {
            "value": metric,
            "peer_mean": 0.0,
            "peer_std": 0.0,
            "z_score": 0.0,
            "percentile": 50.0,
            "relative_strength": 0.0,
        }

    peer_mean = float(np.mean(peer_arr))
    peer_std = float(np.std(peer_arr)) if len(peer_arr) > 1 else 1.0

    z_score = (metric - peer_mean) / (peer_std + 1e-10)
    percentile = float(np.mean(peer_arr <= metric) * 100)
    relative_strength = metric - peer_mean

    return {
        "value": round(metric, 4),
        "peer_mean": round(peer_mean, 4),
        "peer_std": round(peer_std, 4),
        "z_score": round(z_score, 4),
        "percentile": round(percentile, 2),
        "relative_strength": round(relative_strength, 4),
    }


# ─── Fetch Peer Data ────────────────────────────────────────────────


def get_peer_tickers(ticker: str) -> list:
    """
    Get peer company tickers for comparison.

    Uses predefined mappings from config, with a fallback to
    yfinance sector-based lookup.

    Parameters
    ----------
    ticker : str
        Target company ticker.

    Returns
    -------
    list of str
        Peer ticker symbols.
    """
    if ticker in DEFAULT_PEERS:
        return DEFAULT_PEERS[ticker]

    # Fallback: try to find peers via yfinance sector
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        # Return empty if no match — user can extend DEFAULT_PEERS
        print(f"  ⚠ No predefined peers for {ticker} ({sector}/{industry}). "
              f"Add to config.DEFAULT_PEERS for better analysis.")
        return []
    except Exception:
        return []


def fetch_peer_metric(
    tickers: list, metric_fn, metric_name: str = "metric"
) -> dict:
    """
    Fetch a specific metric for multiple peer tickers.

    Parameters
    ----------
    tickers : list of str
        Peer ticker symbols.
    metric_fn : callable
        Function that takes a ticker string and returns a float.
    metric_name : str
        Label for the metric (for logging).

    Returns
    -------
    dict
        Mapping of ticker → metric value.
    """
    results = {}
    for t in tickers:
        try:
            val = metric_fn(t)
            results[t] = val
        except Exception as e:
            print(f"  ⚠ Could not fetch {metric_name} for {t}: {e}")
            results[t] = None
    return results


# ─── Multi-Dimensional Peer Analysis ────────────────────────────────


def compute_peer_analysis(ticker: str, peer_tickers: list = None) -> dict:
    """
    Perform a multi-dimensional peer comparison.

    Compares the target company against peers on:
    - Market capitalisation
    - Revenue growth (trailing)
    - Profit margin
    - Price momentum (YTD)

    Parameters
    ----------
    ticker : str
        Target company ticker.
    peer_tickers : list of str, optional
        Peer tickers. If None, uses defaults from config.

    Returns
    -------
    dict
        Keys: 'peer_score' (composite 0–1), 'dimensions' (per-metric details),
              'peers_used', 'peer_count'.
    """
    if peer_tickers is None:
        peer_tickers = get_peer_tickers(ticker)

    if not peer_tickers:
        return {
            "peer_score": 0.5,
            "signal": "NO_PEERS",
            "dimensions": {},
            "peers_used": [],
            "peer_count": 0,
        }

    print(f"[Peers] Comparing {ticker} against {peer_tickers}")

    dimensions = {}
    scores = []

    # ── 1. Market cap comparison ─────────────────────────────────
    def get_market_cap(t):
        return yf.Ticker(t).info.get("marketCap", 0)

    try:
        target_mcap = get_market_cap(ticker)
        peer_mcaps = fetch_peer_metric(peer_tickers, get_market_cap, "market_cap")
        comp = compute_peer_strength(target_mcap, list(peer_mcaps.values()))
        dimensions["market_cap"] = comp
        scores.append(comp["percentile"] / 100)
    except Exception:
        pass

    # ── 2. Revenue growth comparison ─────────────────────────────
    def get_rev_growth(t):
        tk = yf.Ticker(t)
        rev = tk.quarterly_financials.T
        if "Total Revenue" in rev.columns and len(rev) >= 2:
            r = rev["Total Revenue"].dropna()
            return float((r.iloc[0] - r.iloc[-1]) / abs(r.iloc[-1] + 1e-10))
        return 0.0

    try:
        target_growth = get_rev_growth(ticker)
        peer_growths = fetch_peer_metric(peer_tickers, get_rev_growth, "rev_growth")
        comp = compute_peer_strength(target_growth, list(peer_growths.values()))
        dimensions["revenue_growth"] = comp
        scores.append(comp["percentile"] / 100)
    except Exception:
        pass

    # ── 3. Profit margin comparison ──────────────────────────────
    def get_profit_margin(t):
        return yf.Ticker(t).info.get("profitMargins", 0) or 0

    try:
        target_margin = get_profit_margin(ticker)
        peer_margins = fetch_peer_metric(peer_tickers, get_profit_margin, "profit_margin")
        comp = compute_peer_strength(target_margin, list(peer_margins.values()))
        dimensions["profit_margin"] = comp
        scores.append(comp["percentile"] / 100)
    except Exception:
        pass

    # ── 4. Price momentum (YTD) ──────────────────────────────────
    def get_ytd_return(t):
        data = yf.download(t, period="6mo", progress=False)
        if len(data) >= 2:
            return float(data["Close"].iloc[-1]) / float(data["Close"].iloc[0]) - 1
        return 0.0

    try:
        target_ret = get_ytd_return(ticker)
        peer_rets = fetch_peer_metric(peer_tickers, get_ytd_return, "ytd_return")
        comp = compute_peer_strength(target_ret, list(peer_rets.values()))
        dimensions["price_momentum"] = comp
        scores.append(comp["percentile"] / 100)
    except Exception:
        pass

    # Composite peer score
    peer_score = round(float(np.mean(scores)), 4) if scores else 0.5

    if peer_score >= 0.65:
        signal = "OUTPERFORMING"
    elif peer_score >= 0.35:
        signal = "IN_LINE"
    else:
        signal = "UNDERPERFORMING"

    return {
        "peer_score": peer_score,
        "signal": signal,
        "dimensions": dimensions,
        "peers_used": peer_tickers,
        "peer_count": len(peer_tickers),
    }


if __name__ == "__main__":
    import json

    result = compute_peer_analysis("AAPL")
    # Convert for JSON serialisation
    print(json.dumps(result, indent=2, default=str))
