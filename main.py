"""
main.py — End-to-End Corporate Intelligence Scoring Pipeline
=============================================================
Orchestrates the full analysis flow:
  1. Data ingestion (yfinance)
  2. Fraud probability computation (forensic engine)
  3. Fraud-adjusted financial correction
  4. Market confidence scoring
  5. News risk scoring
  6. Peer comparison analysis
  7. Enhanced FASS true scalability scoring
  8. Executive report generation

Usage:
    python main.py --ticker AAPL
    python main.py --ticker TSLA --period 2y
"""

import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.data_pipeline import fetch_all_data
from models.fraud_engine import compute_fraud_probability
from models.fraud_adjuster import compute_adjusted_growth, generate_adjustment_report
from models.market_engine import compute_market_confidence
from models.news_engine import compute_news_risk
from models.peer_engine import compute_peer_analysis
from models.fass_core import generate_full_assessment


# ─── Sample News Headlines (placeholder — replace with live API) ────


SAMPLE_HEADLINES = {
    "AAPL": [
        "Apple reports record-breaking quarterly revenue",
        "iPhone sales exceed analyst expectations",
        "Apple faces antitrust scrutiny in European markets",
        "New MacBook Pro receives overwhelmingly positive reviews",
        "Supply chain concerns weigh on Apple stock",
        "Apple announces major investment in AI research",
    ],
    "TSLA": [
        "Tesla deliveries miss Wall Street expectations",
        "Elon Musk announces new Gigafactory plans",
        "Tesla faces increasing competition from Chinese EV makers",
        "Cybertruck production ramps up significantly",
        "Analysts divided on Tesla valuation amid market volatility",
        "Tesla energy storage business shows strong growth",
    ],
    "MSFT": [
        "Microsoft Azure revenue grows 30% year-over-year",
        "Microsoft's AI integration boosts Office 365 adoption",
        "Activision Blizzard acquisition cleared by regulators",
        "Microsoft faces cloud infrastructure capacity constraints",
        "GitHub Copilot reaches milestone user count",
        "Microsoft stock hits new all-time high",
    ],
}


def get_headlines(ticker: str) -> list:
    """Get sample headlines for a ticker (placeholder for live news API)."""
    if ticker in SAMPLE_HEADLINES:
        return SAMPLE_HEADLINES[ticker]
    return [
        f"{ticker} reports quarterly earnings",
        f"Analysts maintain neutral outlook on {ticker}",
        f"{ticker} announces strategic partnership",
        f"Market volatility impacts {ticker} stock price",
    ]


# ─── Main Pipeline ──────────────────────────────────────────────────


def run_pipeline(ticker: str, period: str = "1y") -> dict:
    """
    Execute the full corporate intelligence pipeline.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    period : str
        Stock data lookback period (default '1y').

    Returns
    -------
    dict
        Complete assessment report.
    """
    print(f"\n{'='*60}")
    print(f"  CORPORATE INTELLIGENCE AI — {ticker}")
    print(f"{'='*60}\n")

    # ── Step 1: Data Ingestion ───────────────────────────────────
    print("[1/6] Fetching financial data...")
    data = fetch_all_data(ticker, period=period)

    # ── Step 2: Fraud Analysis ───────────────────────────────────
    print("\n[2/6] Running forensic fraud analysis...")
    fraud_result = compute_fraud_probability(data["financials"])
    print(f"  → Fraud probability: {fraud_result['fraud_probability']:.1%} "
          f"({fraud_result['risk_level']})")

    # ── Step 3: Fraud Adjustment ─────────────────────────────────
    print("\n[3/6] Applying fraud-adjusted corrections...")
    adjustment_result = compute_adjusted_growth(
        data["financials"], fraud_result["fraud_probability"]
    )
    print(f"  → Raw growth: {adjustment_result['raw_growth']:.2%}")
    print(f"  → Adjusted growth: {adjustment_result['adjusted_growth']:.2%}")
    print(f"  → Trust factor: {adjustment_result['trust_factor']:.2%}")

    # ── Step 4: Market Analysis ──────────────────────────────────
    print("\n[4/6] Computing market confidence...")
    market_result = compute_market_confidence(data["stock"])
    print(f"  → Market confidence: {market_result['market_confidence']:.1%} "
          f"({market_result['market_signal']})")
    print(f"  → 30d momentum: {market_result['momentum_30d']:.2%}")
    print(f"  → Max drawdown: {market_result['max_drawdown']:.2%}")

    # ── Step 5: News Risk ────────────────────────────────────────
    print("\n[5/6] Analysing news sentiment...")
    headlines = get_headlines(ticker)
    news_result = compute_news_risk(headlines)
    print(f"  → News risk: {news_result['news_risk']:.1%} "
          f"({news_result['risk_level']})")
    print(f"  → Sentiment: {news_result['mean_sentiment']:.3f}")
    print(f"  → Headlines analysed: {news_result['headline_count']}")

    # ── Step 6: Peer Analysis ────────────────────────────────────
    print("\n[6/6] Running peer comparison...")
    peer_result = compute_peer_analysis(ticker)
    print(f"  → Peer score: {peer_result['peer_score']:.1%} "
          f"({peer_result['signal']})")
    print(f"  → Peers compared: {peer_result['peer_count']}")

    # ── Final Scoring ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Computing True Scalability Score...")
    print("─" * 60)

    assessment = generate_full_assessment(
        ticker=ticker,
        company_info=data["info"],
        fraud_result=fraud_result,
        adjustment_result=adjustment_result,
        market_result=market_result,
        news_result=news_result,
        peer_result=peer_result,
    )

    # Print executive summary
    print(f"\n{assessment['executive_summary']}")

    return assessment


def main():
    parser = argparse.ArgumentParser(
        description="Corporate Integrity & Growth Intelligence AI"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to analyse (default: AAPL)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Stock data lookback period (default: 1y)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full results as JSON",
    )
    args = parser.parse_args()

    assessment = run_pipeline(args.ticker, args.period)

    if args.json:
        # Serialise for export (remove non-serialisable objects)
        clean = {
            "ticker": assessment["ticker"],
            "company": assessment["company"],
            "score": assessment["score"],
            "engines": assessment["engines"],
        }
        print("\n" + json.dumps(clean, indent=2, default=str))


if __name__ == "__main__":
    main()
