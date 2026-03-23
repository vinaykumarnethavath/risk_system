import sys
import os
import pandas as pd

# Put project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.data_pipeline import fetch_all_data
from models.fraud_engine import compute_fraud_probability
from models.fraud_adjuster import compute_adjusted_growth
from models.market_engine import compute_market_confidence
from models.news_engine import compute_news_risk
from models.peer_engine import compute_peer_analysis
from models.fass_core import generate_full_assessment, batch_score

def run_batch():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "NFLX", 
        "JPM", "BAC", "WFC", 
        "JNJ", "PFE", "MRK", 
        "XOM", "CVX", 
        "KO", "PEP", 
        "NKE", "SBUX"
    ]
    
    print(f"Starting batch run for {len(tickers)} companies...")
    assessments = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
        try:
            # Run pipeline steps quietly (ignore prints if possible, or just let them go)
            data = fetch_all_data(ticker, period="1y")
            fraud_result = compute_fraud_probability(data["financials"])
            adjustment_result = compute_adjusted_growth(data["financials"], fraud_result["fraud_probability"])
            market_result = compute_market_confidence(data["stock"])
            
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
            assessments.append(assessment)
            print(f"  -> Score: {assessment['score']['true_scalability_score']} ({assessment['score']['grade']})")
        except Exception as e:
            print(f"  -> FAILED processing {ticker}: {e}")

    print("\nCreating batch summary...")
    df = batch_score(assessments)
    df.to_csv("batch_results_20.csv", index=False)
    print("Exported batch_results_20.csv")
    
    # Print nice table
    print("\n" + "="*80)
    print("  BATCH RESULTS SUMMARY (20 Companies)")
    print("="*80)
    print(df[["ticker", "score", "grade", "fraud_risk", "market_confidence"]].to_string())
    print("="*80)

if __name__ == "__main__":
    # Force UTF-8 reconfigure just in case
    if sys.stdout.encoding.lower() != 'utf-8':
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass
    run_batch()
