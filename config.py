"""
config.py — Central configuration for Corporate Intelligence AI platform.
"""

# ─── Scoring Weights ─────────────────────────────────────────────────
WEIGHTS = {
    "adjusted_growth": 0.40,
    "fraud_penalty": 0.30,
    "news_risk": 0.20,
    "market_confidence": 0.30,
}

# ─── Data Defaults ───────────────────────────────────────────────────
DEFAULT_STOCK_PERIOD = "1y"
DEFAULT_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META"],
    "TSLA": ["F", "GM", "RIVN"],
    "AMZN": ["WMT", "SHOP", "BABA"],
}

# ─── Fraud Engine ────────────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.1
BENFORD_THRESHOLD = 0.25

# ─── News Engine ─────────────────────────────────────────────────────
MAX_NEWS_HEADLINES = 20

# ─── GenAI ───────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4"
