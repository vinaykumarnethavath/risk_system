# Corporate Integrity & Growth Intelligence AI

A comprehensive AI-powered platform for corporate risk assessment, fraud detection, and growth intelligence scoring.

## Architecture

```
corporate_intel_ai/
│
├── config.py                  # Configuration and scoring weights
├── main.py                    # End-to-end orchestration pipeline
│
├── data/                      # Raw and processed datasets
│
├── legacy/
│   ├── fass/                  # Legacy FASS (Fraud-Adjusted Scalability Scoring)
│   └── hybrid_risk/           # Legacy Hybrid Risk Scoring forensic module
│
├── models/
│   ├── fraud_engine.py        # Unified forensic fraud probability engine
│   ├── fraud_adjuster.py      # Fraud-adjusted financial correction layer
│   ├── market_engine.py       # Market confidence scoring engine
│   ├── news_engine.py         # News intelligence risk scoring module
│   ├── peer_engine.py         # Peer-relative performance analysis
│   └── fass_core.py           # Enhanced FASS multi-signal scoring model
│
├── pipelines/
│   └── data_pipeline.py       # Financial and stock data ingestion
│
├── services/
│   ├── explanation_engine.py  # GenAI analyst-style reporting
│   └── dashboard.py           # Interactive Streamlit dashboard
│
├── notebooks/
│   └── backtest.ipynb         # Historical backtesting framework
│
└── tests/
    ├── test_fraud.py
    ├── test_market.py
    └── test_news.py
```

## Key Features

- **Forensic Fraud Detection** — Benford's Law, Isolation Forest, statistical anomaly scoring
- **Fraud-Adjusted Financials** — Trust-weighted financial corrections (innovation layer)
- **Multi-Signal Intelligence** — Combines fraud, market momentum, news sentiment, and peer analysis
- **Enhanced FASS Scoring** — True scalability scoring with forensic adjustments
- **GenAI Explanations** — Analyst-style narrative reports powered by LLMs
- **Historical Backtesting** — Validated against Wirecard, SVB, and other corporate failures
- **Interactive Dashboard** — Radar charts, risk scores, and drill-down analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --ticker AAPL
```

## License

MIT
