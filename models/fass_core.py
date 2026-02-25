"""
fass_core.py — Enhanced FASS (Fraud-Adjusted Scalability Scoring) Core
======================================================================
The central scoring model that integrates all signal engines:
  - Fraud probability (forensic engine)
  - Fraud-adjusted financials (adjustment layer)
  - Market confidence (stock trends)
  - News risk (sentiment intelligence)
  - Peer strength (relative performance)

Produces a normalised True Scalability Score (0–100) with
detailed component breakdown and risk classification.
"""

import numpy as np
import pandas as pd

from config import WEIGHTS


# ─── Core Scoring Function ──────────────────────────────────────────


def compute_true_scalability(
    adjusted_growth: float,
    fraud_prob: float,
    news_risk: float,
    market_confidence: float,
    peer_score: float = 0.5,
) -> dict:
    """
    Compute the True Scalability Score by combining all signals.

    Formula:
        raw = (w_growth × adjusted_growth)
            − (w_fraud × fraud_prob)
            − (w_news × news_risk)
            + (w_market × market_confidence)
            + (w_peer × peer_score − 0.5)

    The raw score is then normalised to 0–100 via sigmoid.

    Parameters
    ----------
    adjusted_growth : float
        Fraud-adjusted revenue growth rate.
    fraud_prob : float
        Fraud probability (0–1).
    news_risk : float
        News-based risk score (0–1).
    market_confidence : float
        Market confidence score (0–1).
    peer_score : float
        Peer relative performance score (0–1).

    Returns
    -------
    dict
        Keys: 'true_scalability_score', 'grade', 'components',
              'raw_score', 'signal_contributions'.
    """
    w = WEIGHTS

    # Individual contributions
    growth_contrib = w["adjusted_growth"] * adjusted_growth
    fraud_penalty = w["fraud_penalty"] * fraud_prob
    news_penalty = w["news_risk"] * news_risk
    market_bonus = w["market_confidence"] * market_confidence
    peer_bonus = 0.15 * (peer_score - 0.5)  # Centred around neutral

    # Raw composite
    raw = growth_contrib - fraud_penalty - news_penalty + market_bonus + peer_bonus

    # Normalise to 0–100 using sigmoid
    normalised = 1 / (1 + np.exp(-4 * raw))
    score = round(float(normalised * 100), 2)

    # Grade assignment
    grade = _assign_grade(score)

    # Signal contribution breakdown
    contributions = {
        "growth_contribution": round(float(growth_contrib), 4),
        "fraud_penalty": round(float(-fraud_penalty), 4),
        "news_penalty": round(float(-news_penalty), 4),
        "market_bonus": round(float(market_bonus), 4),
        "peer_bonus": round(float(peer_bonus), 4),
    }

    return {
        "true_scalability_score": score,
        "grade": grade,
        "raw_score": round(float(raw), 4),
        "components": {
            "adjusted_growth": round(adjusted_growth, 4),
            "fraud_probability": round(fraud_prob, 4),
            "news_risk": round(news_risk, 4),
            "market_confidence": round(market_confidence, 4),
            "peer_score": round(peer_score, 4),
        },
        "signal_contributions": contributions,
    }


# ─── Grade Assignment ───────────────────────────────────────────────


def _assign_grade(score: float) -> str:
    """
    Assign a letter grade based on the scalability score.

    Score ranges:
        85–100 → A+ (Exceptional)
        75–84  → A  (Strong)
        65–74  → B+ (Above Average)
        55–64  → B  (Average)
        45–54  → C  (Below Average)
        35–44  → D  (Weak)
         0–34  → F  (Critical Risk)
    """
    if score >= 85:
        return "A+"
    elif score >= 75:
        return "A"
    elif score >= 65:
        return "B+"
    elif score >= 55:
        return "B"
    elif score >= 45:
        return "C"
    elif score >= 35:
        return "D"
    else:
        return "F"


# ─── Full Assessment Report ─────────────────────────────────────────


def generate_full_assessment(
    ticker: str,
    company_info: dict,
    fraud_result: dict,
    adjustment_result: dict,
    market_result: dict,
    news_result: dict,
    peer_result: dict,
) -> dict:
    """
    Generate a comprehensive corporate intelligence assessment.

    Aggregates all engine outputs into a single structured report.

    Parameters
    ----------
    ticker : str
        Company ticker symbol.
    company_info : dict
        Company metadata from data pipeline.
    fraud_result : dict
        Output from fraud_engine.compute_fraud_probability().
    adjustment_result : dict
        Output from fraud_adjuster.compute_adjusted_growth().
    market_result : dict
        Output from market_engine.compute_market_confidence().
    news_result : dict
        Output from news_engine.compute_news_risk().
    peer_result : dict
        Output from peer_engine.compute_peer_analysis().

    Returns
    -------
    dict
        Full assessment with score, grade, all components, and
        executive summary.
    """
    # Extract key inputs
    adjusted_growth = adjustment_result.get("adjusted_growth", 0.0)
    fraud_prob = fraud_result.get("fraud_probability", 0.0)
    news_risk = news_result.get("news_risk", 0.5)
    market_confidence = market_result.get("market_confidence", 0.5)
    peer_score = peer_result.get("peer_score", 0.5)

    # Compute core score
    score_result = compute_true_scalability(
        adjusted_growth=adjusted_growth,
        fraud_prob=fraud_prob,
        news_risk=news_risk,
        market_confidence=market_confidence,
        peer_score=peer_score,
    )

    # Build executive summary
    summary = _build_executive_summary(
        ticker, company_info, score_result, fraud_result,
        market_result, news_result, peer_result
    )

    return {
        "ticker": ticker,
        "company": company_info,
        "score": score_result,
        "engines": {
            "fraud": fraud_result,
            "adjustment": adjustment_result,
            "market": market_result,
            "news": {k: v for k, v in news_result.items() if k != "details"},
            "peer": {k: v for k, v in peer_result.items() if k != "dimensions"},
        },
        "executive_summary": summary,
    }


def _build_executive_summary(
    ticker: str,
    info: dict,
    score: dict,
    fraud: dict,
    market: dict,
    news: dict,
    peer: dict,
) -> str:
    """Generate a human-readable executive summary."""
    name = info.get("name", ticker)
    grade = score["grade"]
    tss = score["true_scalability_score"]

    lines = [
        f"{'='*60}",
        f"  CORPORATE INTELLIGENCE REPORT: {name} ({ticker})",
        f"{'='*60}",
        f"",
        f"  True Scalability Score : {tss}/100  (Grade: {grade})",
        f"",
        f"  ┌─ Fraud Risk       : {fraud.get('risk_level', 'N/A'):>15}  "
        f"({fraud.get('fraud_probability', 0):.1%})",
        f"  ├─ Market Signal    : {market.get('market_signal', 'N/A'):>15}  "
        f"({market.get('market_confidence', 0):.1%})",
        f"  ├─ News Sentiment   : {news.get('risk_level', 'N/A'):>15}  "
        f"(risk: {news.get('news_risk', 0):.1%})",
        f"  └─ Peer Position    : {peer.get('signal', 'N/A'):>15}  "
        f"({peer.get('peer_score', 0):.1%})",
        f"",
        f"  Sector: {info.get('sector', 'N/A')}  |  "
        f"Industry: {info.get('industry', 'N/A')}",
        f"{'='*60}",
    ]

    return "\n".join(lines)


# ─── Batch Scoring ──────────────────────────────────────────────────


def batch_score(assessments: list) -> pd.DataFrame:
    """
    Convert a list of assessment dicts into a comparison DataFrame.

    Parameters
    ----------
    assessments : list of dict
        Each from generate_full_assessment().

    Returns
    -------
    pd.DataFrame
        One row per company with key scores.
    """
    rows = []
    for a in assessments:
        rows.append({
            "ticker": a["ticker"],
            "company": a["company"].get("name", a["ticker"]),
            "score": a["score"]["true_scalability_score"],
            "grade": a["score"]["grade"],
            "fraud_risk": a["engines"]["fraud"].get("fraud_probability", 0),
            "market_confidence": a["engines"]["market"].get("market_confidence", 0),
            "news_risk": a["engines"]["news"].get("news_risk", 0.5),
            "peer_score": a["engines"]["peer"].get("peer_score", 0.5),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick demo with mock data
    result = compute_true_scalability(
        adjusted_growth=0.15,
        fraud_prob=0.2,
        news_risk=0.35,
        market_confidence=0.7,
        peer_score=0.6,
    )
    print(f"True Scalability Score: {result['true_scalability_score']}/100")
    print(f"Grade: {result['grade']}")
    print(f"Components: {result['components']}")
    print(f"Contributions: {result['signal_contributions']}")
