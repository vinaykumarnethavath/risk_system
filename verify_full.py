import sys
import os

# Put project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.data_pipeline import fetch_all_data
from models.fraud_engine import compute_fraud_probability
from models.fraud_adjuster import compute_adjusted_growth
from models.market_engine import compute_market_confidence
from models.news_engine import compute_news_risk
from models.peer_engine import compute_peer_analysis
from models.fass_core import generate_full_assessment
from services.explanation_engine import generate_report

def verify_ticker(ticker):
    print(f"\n{'='*60}")
    print(f"  VERIFYING {ticker}")
    print(f"{'='*60}")
    
    # Run pipeline steps
    data = fetch_all_data(ticker, period="1y")
    fraud_result = compute_fraud_probability(data["financials"])
    adjustment_result = compute_adjusted_growth(data["financials"], fraud_result["fraud_probability"])
    market_result = compute_market_confidence(data["stock"])
    
    # News fallback
    from main import get_headlines
    headlines = get_headlines(ticker, data["info"].get("longName", None))
    news_result = compute_news_risk(headlines)
    
    peer_result = compute_peer_analysis(ticker)
    
    assessment = generate_full_assessment(
        ticker=ticker,
        company_info=data["info"],
        fraud_result=fraud_result,
        adjustment_result=adjustment_result,
        market_result=market_result,
        news_result=news_result,
        peer_result=peer_result
    )
    
    print("\n--- Executive Summary (fass_core) ---")
    print(assessment["executive_summary"])
    
    print("\n--- Narrative Report (explanation_engine) ---")
    report = generate_report(assessment, use_genai=False)
    print(report)

verify_ticker("TSLA")
