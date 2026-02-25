"""
news_engine.py — News Intelligence Risk Scoring Module
======================================================
Computes news-based risk signals using VADER sentiment analysis.
Supports both headline-based scoring and full article analysis.

Provides:
  - Individual headline sentiment scoring
  - Aggregate news risk computation
  - Temporal sentiment trend analysis
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import MAX_NEWS_HEADLINES


# ─── Sentiment Analyser ─────────────────────────────────────────────


_analyzer = SentimentIntensityAnalyzer()


def score_headline(headline: str) -> dict:
    """
    Score a single headline using VADER sentiment analysis.

    Parameters
    ----------
    headline : str
        News headline text.

    Returns
    -------
    dict
        Keys: 'compound', 'positive', 'negative', 'neutral'.
    """
    scores = _analyzer.polarity_scores(headline)
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
    }


def score_headlines(headlines: list) -> pd.DataFrame:
    """
    Score multiple headlines and return a structured DataFrame.

    Parameters
    ----------
    headlines : list of str
        News headlines.

    Returns
    -------
    pd.DataFrame
        Columns: headline, compound, positive, negative, neutral.
    """
    rows = []
    for h in headlines[:MAX_NEWS_HEADLINES]:
        s = score_headline(h)
        s["headline"] = h
        rows.append(s)
    return pd.DataFrame(rows)


# ─── News Risk Computation ──────────────────────────────────────────


def compute_news_risk(headlines: list) -> dict:
    """
    Compute an aggregate news risk score from a list of headlines.

    Risk formula:
        risk = 1 − (mean_compound + 1) / 2

    So compound = +1 (very positive) → risk = 0
       compound = −1 (very negative) → risk = 1
       compound =  0 (neutral)       → risk = 0.5

    Parameters
    ----------
    headlines : list of str
        News headlines about the company.

    Returns
    -------
    dict
        Keys:
        - 'news_risk': float (0–1)
        - 'mean_sentiment': float (-1 to +1)
        - 'positive_pct': float, fraction of positive headlines
        - 'negative_pct': float, fraction of negative headlines
        - 'headline_count': int
        - 'risk_level': str ('LOW' / 'MEDIUM' / 'HIGH')
        - 'details': pd.DataFrame with per-headline scores
    """
    if not headlines:
        return {
            "news_risk": 0.5,
            "mean_sentiment": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "headline_count": 0,
            "risk_level": "NEUTRAL",
            "details": pd.DataFrame(),
        }

    details = score_headlines(headlines)
    compounds = details["compound"].values

    mean_sentiment = float(np.mean(compounds))
    news_risk = round(1 - (mean_sentiment + 1) / 2, 4)

    positive_pct = round(float((compounds > 0.05).mean()), 4)
    negative_pct = round(float((compounds < -0.05).mean()), 4)

    # Risk classification
    if news_risk >= 0.65:
        level = "HIGH"
    elif news_risk >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "news_risk": news_risk,
        "mean_sentiment": round(mean_sentiment, 4),
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "headline_count": len(details),
        "risk_level": level,
        "details": details,
    }


# ─── Temporal Sentiment Trend ───────────────────────────────────────


def compute_sentiment_trend(
    headlines: list, timestamps: list = None
) -> dict:
    """
    Analyse sentiment trend over time (if timestamps provided).

    Parameters
    ----------
    headlines : list of str
        Headlines ordered chronologically.
    timestamps : list, optional
        Corresponding timestamps (parseable by pd.to_datetime).

    Returns
    -------
    dict
        Keys: 'trend_direction', 'trend_slope',
              'recent_sentiment', 'older_sentiment'.
    """
    if len(headlines) < 4:
        return {
            "trend_direction": "INSUFFICIENT_DATA",
            "trend_slope": 0.0,
            "recent_sentiment": 0.0,
            "older_sentiment": 0.0,
        }

    scores = [score_headline(h)["compound"] for h in headlines]
    mid = len(scores) // 2

    older = float(np.mean(scores[:mid]))
    recent = float(np.mean(scores[mid:]))
    slope = recent - older

    if slope > 0.1:
        direction = "IMPROVING"
    elif slope < -0.1:
        direction = "DETERIORATING"
    else:
        direction = "STABLE"

    return {
        "trend_direction": direction,
        "trend_slope": round(slope, 4),
        "recent_sentiment": round(recent, 4),
        "older_sentiment": round(older, 4),
    }


if __name__ == "__main__":
    import json

    sample_headlines = [
        "Apple reports record-breaking quarterly revenue",
        "iPhone sales exceed analyst expectations",
        "Apple faces antitrust scrutiny in European markets",
        "New MacBook Pro receives overwhelmingly positive reviews",
        "Supply chain concerns weigh on Apple stock",
        "Apple announces major investment in AI research",
        "Analysts upgrade Apple stock to outperform",
        "Apple hit with patent infringement lawsuit",
    ]

    result = compute_news_risk(sample_headlines)
    # Remove DataFrame for JSON printing
    details = result.pop("details")
    print(json.dumps(result, indent=2))
    print("\nPer-headline details:")
    print(details.to_string(index=False))

    trend = compute_sentiment_trend(sample_headlines)
    print("\nSentiment trend:")
    print(json.dumps(trend, indent=2))
