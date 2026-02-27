"""
dashboard.py — Interactive Corporate Risk Intelligence Dashboard
================================================================
Streamlit-based dashboard for visualising corporate intelligence
scores, radar charts, and detailed engine breakdowns.

Usage:
    streamlit run services/dashboard.py
"""

import sys
import os
import json

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipelines.data_pipeline import fetch_all_data
from models.fraud_engine import compute_fraud_probability
from models.fraud_adjuster import compute_adjusted_growth, generate_adjustment_report
from models.market_engine import compute_market_confidence
from models.news_engine import compute_news_risk
from models.peer_engine import compute_peer_analysis
from models.fass_core import generate_full_assessment, compute_true_scalability
from services.explanation_engine import generate_report


# ─── Page Configuration ─────────────────────────────────────────────


st.set_page_config(
    page_title="Corporate Intelligence AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom Styling ──────────────────────────────────────────────────


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.3rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────


st.sidebar.markdown("## 🏢 Corporate Intelligence AI")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Analysis Period", ["6mo", "1y", "2y", "5y"], index=1)

analyse_btn = st.sidebar.button("🔍 Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Tickers")
quick_tickers = st.sidebar.columns(3)
for i, t in enumerate(["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]):
    if quick_tickers[i % 3].button(t, key=f"quick_{t}"):
        ticker = t
        analyse_btn = True

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using\n"
    "forensic analytics, ML,\n"
    "and multi-signal intelligence"
)


# ─── Helper Functions ───────────────────────────────────────────────


def get_grade_color(grade: str) -> str:
    """Return color for grade badge."""
    colors = {
        "A+": "#00c853", "A": "#2e7d32", "B+": "#558b2f",
        "B": "#f9a825", "C": "#ef6c00", "D": "#d84315", "F": "#b71c1c",
    }
    return colors.get(grade, "#757575")


def create_radar_chart(components: dict) -> go.Figure:
    """Create a radar chart of scoring components."""
    categories = ["Growth", "Fraud Safety", "News Safety", "Market", "Peer"]
    values = [
        max(0, min(1, components.get("adjusted_growth", 0) + 0.5)),
        1 - components.get("fraud_probability", 0),
        1 - components.get("news_risk", 0.5),
        components.get("market_confidence", 0.5),
        components.get("peer_score", 0.5),
    ]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.25)",
        line=dict(color="#667eea", width=2),
        marker=dict(size=8, color="#667eea"),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """Create a gauge chart for a single metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 24}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0.05)",
            borderwidth=0,
            steps=[
                {"range": [0, 33], "color": "rgba(76, 175, 80, 0.1)"},
                {"range": [33, 66], "color": "rgba(255, 193, 7, 0.1)"},
                {"range": [66, 100], "color": "rgba(244, 67, 54, 0.1)"},
            ],
        ),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── Main Content ───────────────────────────────────────────────────


st.markdown('<p class="main-header">🏢 Corporate Intelligence AI</p>', unsafe_allow_html=True)
st.markdown("*Multi-signal corporate risk assessment powered by forensic analytics and AI*")

if analyse_btn:
    # Progress bar
    progress = st.progress(0, text="Initialising analysis...")

    try:
        # Step 1: Data
        progress.progress(10, text="📊 Fetching financial data...")
        data = fetch_all_data(ticker, period=period)

        # Step 2: Fraud
        progress.progress(25, text="🔍 Running forensic fraud analysis...")
        fraud_result = compute_fraud_probability(data["financials"])

        # Step 3: Adjustment
        progress.progress(40, text="⚖️ Computing fraud-adjusted financials...")
        adjustment_result = compute_adjusted_growth(
            data["financials"], fraud_result["fraud_probability"]
        )

        # Step 4: Market
        progress.progress(55, text="📈 Analysing market signals...")
        market_result = compute_market_confidence(data["stock"])

        # Step 5: News
        progress.progress(70, text="📰 Scoring news sentiment...")
        from main import get_headlines
        headlines = get_headlines(ticker)
        news_result = compute_news_risk(headlines)

        # Step 6: Peers
        progress.progress(85, text="👥 Running peer comparison...")
        peer_result = compute_peer_analysis(ticker)

        # Step 7: Final Score
        progress.progress(95, text="🧠 Computing True Scalability Score...")
        assessment = generate_full_assessment(
            ticker=ticker,
            company_info=data["info"],
            fraud_result=fraud_result,
            adjustment_result=adjustment_result,
            market_result=market_result,
            news_result=news_result,
            peer_result=peer_result,
        )

        progress.progress(100, text="✅ Analysis complete!")

        score = assessment["score"]
        info = assessment["company"]
        tss = score["true_scalability_score"]
        grade = score["grade"]

        # ── Header Row ───────────────────────────────────────────
        st.markdown("---")
        col_info, col_score = st.columns([2, 1])

        with col_info:
            st.markdown(f"### {info.get('name', ticker)} ({ticker})")
            st.markdown(
                f"**Sector:** {info.get('sector', 'N/A')} · "
                f"**Industry:** {info.get('industry', 'N/A')} · "
                f"**Market Cap:** ${info.get('market_cap', 0):,.0f}"
            )

        with col_score:
            grade_color = get_grade_color(grade)
            st.markdown(
                f'<div class="score-card">'
                f'<div style="font-size: 0.9rem; opacity: 0.8;">TRUE SCALABILITY SCORE</div>'
                f'<div class="score-value" style="color: {grade_color}">{tss}</div>'
                f'<div class="grade-badge" style="background: {grade_color}; color: white;">'
                f'Grade {grade}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Radar Chart + Signals ────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Signal Analysis")

        col_radar, col_signals = st.columns([1, 1])

        with col_radar:
            radar = create_radar_chart(score["components"])
            st.plotly_chart(radar, use_container_width=True)

        with col_signals:
            # Fraud
            fraud_color = "#b71c1c" if fraud_result["fraud_probability"] > 0.5 else (
                "#ef6c00" if fraud_result["fraud_probability"] > 0.3 else "#2e7d32"
            )
            st.markdown(
                f'<div class="metric-card" style="border-color: {fraud_color}">'
                f'<b>🔍 Fraud Risk:</b> {fraud_result["fraud_probability"]:.1%} '
                f'({fraud_result["risk_level"]})</div>',
                unsafe_allow_html=True,
            )

            # Market
            mkt_color = "#2e7d32" if market_result["market_confidence"] > 0.6 else (
                "#ef6c00" if market_result["market_confidence"] > 0.4 else "#b71c1c"
            )
            st.markdown(
                f'<div class="metric-card" style="border-color: {mkt_color}">'
                f'<b>📈 Market Signal:</b> {market_result["market_confidence"]:.1%} '
                f'({market_result["market_signal"]})</div>',
                unsafe_allow_html=True,
            )

            # News
            news_color = "#b71c1c" if news_result["news_risk"] > 0.6 else (
                "#ef6c00" if news_result["news_risk"] > 0.4 else "#2e7d32"
            )
            st.markdown(
                f'<div class="metric-card" style="border-color: {news_color}">'
                f'<b>📰 News Risk:</b> {news_result["news_risk"]:.1%} '
                f'({news_result["risk_level"]})</div>',
                unsafe_allow_html=True,
            )

            # Peer
            peer_color = "#2e7d32" if peer_result["peer_score"] > 0.6 else (
                "#ef6c00" if peer_result["peer_score"] > 0.4 else "#b71c1c"
            )
            st.markdown(
                f'<div class="metric-card" style="border-color: {peer_color}">'
                f'<b>👥 Peer Position:</b> {peer_result["peer_score"]:.1%} '
                f'({peer_result["signal"]})</div>',
                unsafe_allow_html=True,
            )

        # ── Gauge Charts ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Engine Scores")

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(
                create_gauge_chart(fraud_result["fraud_probability"], "Fraud Risk", "#e53935"),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(
                create_gauge_chart(market_result["market_confidence"], "Market", "#43a047"),
                use_container_width=True,
            )
        with g3:
            st.plotly_chart(
                create_gauge_chart(news_result["news_risk"], "News Risk", "#fb8c00"),
                use_container_width=True,
            )
        with g4:
            st.plotly_chart(
                create_gauge_chart(peer_result["peer_score"], "Peer Score", "#1e88e5"),
                use_container_width=True,
            )

        # ── Analyst Report ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📝 Analyst Report")

        report = generate_report(assessment, use_genai=False)
        st.markdown(report)

        # ── Detailed Breakdown (Expandable) ──────────────────────
        st.markdown("---")
        with st.expander("🔬 Detailed Signal Contributions"):
            contrib = score["signal_contributions"]
            contrib_df = pd.DataFrame([
                {"Signal": "Growth Contribution", "Value": contrib["growth_contribution"]},
                {"Signal": "Fraud Penalty", "Value": contrib["fraud_penalty"]},
                {"Signal": "News Penalty", "Value": contrib["news_penalty"]},
                {"Signal": "Market Bonus", "Value": contrib["market_bonus"]},
                {"Signal": "Peer Bonus", "Value": contrib["peer_bonus"]},
            ])
            fig_bar = px.bar(
                contrib_df, x="Signal", y="Value",
                color="Value",
                color_continuous_scale=["#e53935", "#ffeb3b", "#43a047"],
                title="Signal Contributions to Final Score",
            )
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("📊 Raw Assessment Data (JSON)"):
            clean = {
                "ticker": assessment["ticker"],
                "company": assessment["company"],
                "score": assessment["score"],
                "engines": assessment["engines"],
            }
            st.json(clean)

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

else:
    # Landing state
    st.markdown("---")
    st.info(
        "👈 Enter a **stock ticker** in the sidebar and click "
        "**Run Analysis** to start."
    )

    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔍 Forensic Analysis")
        st.markdown(
            "Benford's Law, statistical anomalies, "
            "and Isolation Forest detect financial manipulation."
        )

    with col2:
        st.markdown("#### 📊 Multi-Signal Intelligence")
        st.markdown(
            "Market momentum, news sentiment, and peer "
            "comparison provide holistic risk assessment."
        )

    with col3:
        st.markdown("#### 🧠 AI-Powered Insights")
        st.markdown(
            "GenAI generates analyst-style reports "
            "with actionable investment recommendations."
        )
