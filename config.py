"""
config.py — Central Configuration for Corporate Intelligence AI
================================================================
All scoring weights, thresholds, and parameters are configurable
here. Adjust these to tune the scoring model without touching
engine code.

Supports environment variable overrides for deployment flexibility.
"""

import os

# ─── Scoring Weights (must sum contextually, not necessarily to 1) ──
# These control how much each signal contributes to the final
# True Scalability Score in fass_core.py.
#
# Formula:
#   raw = (adjusted_growth × W_GROWTH)
#       − (fraud_prob × W_FRAUD)
#       − (news_risk × W_NEWS)
#       + (market_confidence × W_MARKET)
#       + (peer_score − 0.5) × W_PEER

WEIGHTS = {
    "adjusted_growth": float(os.environ.get("WEIGHT_GROWTH", 0.40)),
    "fraud_penalty": float(os.environ.get("WEIGHT_FRAUD", 0.30)),
    "news_risk": float(os.environ.get("WEIGHT_NEWS", 0.20)),
    "market_confidence": float(os.environ.get("WEIGHT_MARKET", 0.30)),
    "peer_bonus": float(os.environ.get("WEIGHT_PEER", 0.15)),
}


# ─── Scoring Normalisation ──────────────────────────────────────────

# Sigmoid steepness for final score normalisation (higher = sharper)
SIGMOID_STEEPNESS = float(os.environ.get("SIGMOID_STEEPNESS", 4.0))

# Grade boundaries (score thresholds for letter grades)
GRADE_BOUNDARIES = {
    "A+": 85,
    "A": 75,
    "B+": 65,
    "B": 55,
    "C": 45,
    "D": 35,
    # Below 35 = F
}


# ─── Data Defaults ──────────────────────────────────────────────────

DEFAULT_STOCK_PERIOD = os.environ.get("STOCK_PERIOD", "1y")

DEFAULT_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META"],
    "TSLA": ["F", "GM", "RIVN"],
    "AMZN": ["WMT", "SHOP", "BABA"],
    "MSFT": ["AAPL", "GOOGL", "ORCL"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
}


# ─── Fraud Engine Parameters ────────────────────────────────────────

# Isolation Forest contamination (expected fraction of anomalies)
ISOLATION_FOREST_CONTAMINATION = float(
    os.environ.get("ISO_CONTAMINATION", 0.1)
)

# Benford's Law deviation threshold for flagging
BENFORD_THRESHOLD = float(os.environ.get("BENFORD_THRESHOLD", 0.25))

# Fraud signal weights (within the fraud engine)
FRAUD_SIGNAL_WEIGHTS = {
    "benford": float(os.environ.get("FRAUD_W_BENFORD", 0.35)),
    "statistical": float(os.environ.get("FRAUD_W_STATISTICAL", 0.35)),
    "isolation_forest": float(os.environ.get("FRAUD_W_ISOLATION", 0.30)),
}

# Fraud risk level thresholds
FRAUD_RISK_THRESHOLDS = {
    "high": float(os.environ.get("FRAUD_THRESH_HIGH", 0.6)),
    "medium": float(os.environ.get("FRAUD_THRESH_MEDIUM", 0.3)),
}


# ─── News Engine Parameters ─────────────────────────────────────────

MAX_NEWS_HEADLINES = int(os.environ.get("MAX_HEADLINES", 20))

# News risk level thresholds
NEWS_RISK_THRESHOLDS = {
    "high": float(os.environ.get("NEWS_THRESH_HIGH", 0.65)),
    "medium": float(os.environ.get("NEWS_THRESH_MEDIUM", 0.45)),
}


# ─── Market Engine Parameters ───────────────────────────────────────

# Market confidence signal weights
MARKET_SIGNAL_WEIGHTS = {
    "volatility_penalty": float(os.environ.get("MKT_W_VOLATILITY", 0.5)),
    "drawdown_penalty": float(os.environ.get("MKT_W_DRAWDOWN", 0.3)),
    "volume_bonus": float(os.environ.get("MKT_W_VOLUME", 0.1)),
}

# Market confidence level thresholds
MARKET_CONFIDENCE_THRESHOLDS = {
    "bullish": float(os.environ.get("MKT_THRESH_BULLISH", 0.65)),
    "neutral": float(os.environ.get("MKT_THRESH_NEUTRAL", 0.45)),
}


# ─── GenAI Settings ─────────────────────────────────────────────────

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", 0.7))
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", 800))


# ─── Utility ─────────────────────────────────────────────────────────


def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("  Corporate Intelligence AI — Configuration")
    print("=" * 50)
    print(f"  Scoring Weights: {WEIGHTS}")
    print(f"  Sigmoid Steepness: {SIGMOID_STEEPNESS}")
    print(f"  Grade Boundaries: {GRADE_BOUNDARIES}")
    print(f"  Fraud Signals: {FRAUD_SIGNAL_WEIGHTS}")
    print(f"  Fraud Thresholds: {FRAUD_RISK_THRESHOLDS}")
    print(f"  Benford Threshold: {BENFORD_THRESHOLD}")
    print(f"  IsoForest Contamination: {ISOLATION_FOREST_CONTAMINATION}")
    print(f"  News Thresholds: {NEWS_RISK_THRESHOLDS}")
    print(f"  Market Thresholds: {MARKET_CONFIDENCE_THRESHOLDS}")
    print(f"  OpenAI Model: {OPENAI_MODEL}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
