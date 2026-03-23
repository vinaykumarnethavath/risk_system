"""
test_news.py — Unit tests for the News Engine
==============================================
Tests headline sentiment scoring, aggregate news risk,
and temporal sentiment trend analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.news_engine import (
    score_headline,
    score_headlines,
    compute_news_risk,
    compute_sentiment_trend,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def positive_headlines():
    return [
        "Company reports record-breaking quarterly revenue",
        "Stock upgraded to buy by major analysts",
        "New product launch exceeds all expectations",
        "Company wins prestigious innovation award",
        "Outstanding growth drives investor confidence",
    ]


@pytest.fixture
def negative_headlines():
    return [
        "Company faces massive fraud investigation",
        "Stock plunges after disappointing earnings report",
        "CEO arrested on corruption charges",
        "Regulators impose heavy fine for violations",
        "Company announces major layoffs amid losses",
    ]


@pytest.fixture
def mixed_headlines():
    return [
        "Company reports solid but unremarkable earnings",
        "New CEO appointment brings cautious optimism",
        "Market conditions remain uncertain for the sector",
        "Revenue grows slightly despite challenging environment",
    ]


# ─── Single Headline Tests ──────────────────────────────────────────


class TestScoreHeadline:
    def test_returns_dict(self):
        result = score_headline("This is great news!")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = score_headline("Test headline")
        assert "compound" in result
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result

    def test_positive_headline(self):
        result = score_headline("Great revenue exceeds expectations!")
        assert result["compound"] > 0

    def test_negative_headline(self):
        result = score_headline("Company faces devastating fraud scandal and bankruptcy")
        assert result["compound"] < 0

    def test_compound_bounded(self):
        result = score_headline("Some random headline about things")
        assert -1.0 <= result["compound"] <= 1.0

    def test_empty_string(self):
        result = score_headline("")
        assert result["compound"] == 0.0


# ─── Multiple Headlines Tests ───────────────────────────────────────


class TestScoreHeadlines:
    def test_returns_dataframe(self, positive_headlines):
        result = score_headlines(positive_headlines)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, positive_headlines):
        result = score_headlines(positive_headlines)
        assert len(result) == len(positive_headlines)

    def test_has_headline_column(self, positive_headlines):
        result = score_headlines(positive_headlines)
        assert "headline" in result.columns
        assert "compound" in result.columns

    def test_empty_list(self):
        result = score_headlines([])
        assert len(result) == 0


# ─── News Risk Tests ────────────────────────────────────────────────


class TestComputeNewsRisk:
    def test_returns_dict(self, positive_headlines):
        result = compute_news_risk(positive_headlines)
        assert isinstance(result, dict)

    def test_has_required_keys(self, positive_headlines):
        result = compute_news_risk(positive_headlines)
        assert "news_risk" in result
        assert "mean_sentiment" in result
        assert "positive_pct" in result
        assert "negative_pct" in result
        assert "headline_count" in result
        assert "risk_level" in result

    def test_risk_bounded(self, positive_headlines):
        result = compute_news_risk(positive_headlines)
        assert 0.0 <= result["news_risk"] <= 1.0

    def test_positive_news_low_risk(self, positive_headlines):
        result = compute_news_risk(positive_headlines)
        assert result["news_risk"] < 0.5
        assert result["mean_sentiment"] > 0

    def test_negative_news_high_risk(self, negative_headlines):
        result = compute_news_risk(negative_headlines)
        assert result["news_risk"] > 0.5
        assert result["mean_sentiment"] < 0

    def test_risk_level_valid(self, mixed_headlines):
        result = compute_news_risk(mixed_headlines)
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_empty_list_default(self):
        result = compute_news_risk([])
        assert result["news_risk"] == 0.5
        assert result["headline_count"] == 0
        assert result["risk_level"] == "NEUTRAL"

    def test_headline_count_correct(self, positive_headlines):
        result = compute_news_risk(positive_headlines)
        assert result["headline_count"] == len(positive_headlines)

    def test_positive_pct_bounded(self, mixed_headlines):
        result = compute_news_risk(mixed_headlines)
        assert 0.0 <= result["positive_pct"] <= 1.0
        assert 0.0 <= result["negative_pct"] <= 1.0


# ─── Sentiment Trend Tests ──────────────────────────────────────────


class TestComputeSentimentTrend:
    def test_returns_dict(self, positive_headlines):
        result = compute_sentiment_trend(positive_headlines)
        assert isinstance(result, dict)

    def test_has_required_keys(self, positive_headlines):
        result = compute_sentiment_trend(positive_headlines)
        assert "trend_direction" in result
        assert "trend_slope" in result
        assert "recent_sentiment" in result
        assert "older_sentiment" in result

    def test_insufficient_data(self):
        result = compute_sentiment_trend(["Just one headline"])
        assert result["trend_direction"] == "INSUFFICIENT_DATA"

    def test_direction_valid(self, positive_headlines):
        result = compute_sentiment_trend(positive_headlines)
        assert result["trend_direction"] in ["IMPROVING", "STABLE", "DETERIORATING", "INSUFFICIENT_DATA"]

    def test_deteriorating_trend(self):
        headlines = [
            "Excellent performance and record profits",
            "Outstanding growth continues strong momentum",
            "Company faces serious regulatory problems",
            "Stock crashes on fraud allegations and lawsuits",
        ]
        result = compute_sentiment_trend(headlines)
        assert result["trend_slope"] < 0

    def test_improving_trend(self):
        headlines = [
            "Company struggles with declining revenue and losses",
            "Massive layoffs announced amid financial crisis",
            "Strong recovery signals emerge for the company",
            "Record-breaking quarter exceeds all expectations",
        ]
        result = compute_sentiment_trend(headlines)
        assert result["trend_slope"] > 0
