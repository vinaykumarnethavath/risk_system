"""
report_generator.py — Advanced Report Generation System
======================================================
Generates comprehensive corporate intelligence reports in multiple formats:
- Detailed JSON reports
- Executive summaries  
- PDF reports (if dependencies available)
- CSV exports for analysis
- HTML reports for web viewing

Usage:
    from services.report_generator import generate_comprehensive_report
    report = generate_comprehensive_report(assessment, ticker)
"""

import json
import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("⚠️ Jinja2 not available. Install with: pip install jinja2")

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("⚠️ WeasyPrint not available. Install with: pip install weasyprint")


def generate_executive_summary(assessment: Dict[str, Any]) -> str:
    """Generate a comprehensive executive summary."""
    ticker = assessment.get("ticker", "Unknown")
    company = assessment.get("company", {})
    score = assessment.get("score", {})
    engines = assessment.get("engines", {})
    
    company_name = company.get("name", ticker)
    sector = company.get("sector", "Unknown")
    industry = company.get("industry", "Unknown")
    
    tss = score.get("true_scalability_score", 0)
    grade = score.get("grade", "N/A")
    
    fraud_prob = engines.get("fraud", {}).get("fraud_probability", 0)
    fraud_level = engines.get("fraud", {}).get("risk_level", "UNKNOWN")
    
    market_conf = engines.get("market", {}).get("market_confidence", 0)
    market_signal = engines.get("market", {}).get("market_signal", "UNKNOWN")
    
    news_risk = engines.get("news", {}).get("news_risk", 0)
    news_level = engines.get("news", {}).get("risk_level", "UNKNOWN")
    
    peer_score = engines.get("peer", {}).get("peer_score", 0)
    peer_signal = engines.get("peer", {}).get("signal", "UNKNOWN")
    
    # Generate narrative
    summary = f"""
EXECUTIVE SUMMARY: {company_name} ({ticker})

OVERALL ASSESSMENT: {tss:.1f}/100 (Grade: {grade})
{company_name} demonstrates {'strong' if tss >= 60 else 'moderate' if tss >= 40 else 'weak'} corporate intelligence metrics 
with a True Scalability Score of {tss:.1f}/100, placing it in the {grade} category. 
The company operates in the {sector} sector, specifically {industry}.

KEY RISK FACTORS:
• Fraud Risk: {fraud_prob:.1%} ({fraud_level}) - {'Elevated concern' if fraud_prob > 0.2 else 'Normal range'}
• Market Confidence: {market_conf:.1%} ({market_signal}) - {'Bullish' if market_conf > 0.6 else 'Bearish' if market_conf < 0.4 else 'Neutral'} market sentiment
• News Sentiment: {news_risk:.1%} ({news_level}) - {'Positive coverage' if news_risk < 0.3 else 'Mixed/Negative coverage'}
• Peer Performance: {peer_score:.1%} ({peer_signal}) - {'Outperforming' if peer_score > 0.6 else 'Underperforming' if peer_score < 0.4 else 'In-line'}

FINANCIAL HEALTH:
• {'Strong growth trajectory' if tss > 50 else 'Moderate performance' if tss > 30 else 'Concerning trends detected'}
• {'Low fraud indicators suggest reliable financial reporting' if fraud_prob < 0.15 else 'Elevated fraud risk requires closer scrutiny'}
• Market sentiment {'favors the company' if market_conf > 0.5 else 'shows caution' if market_conf > 0.3 else 'indicates significant concern'}

RECOMMENDATIONS:
{'• INVEST: Strong fundamentals and low risk profile suggest investment opportunity' if tss > 60 and fraud_prob < 0.15 else
 '• HOLD: Moderate metrics suggest wait-and-see approach' if 40 <= tss <= 60 else
 '• CAUTION: Elevated risk factors warrant careful consideration'}

NEXT STEPS:
• Monitor quarterly earnings for consistency
• Track peer group performance trends
• Watch for regulatory developments
• Assess market sentiment changes

Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary.strip()


def generate_detailed_json(assessment: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """Generate comprehensive JSON report with additional analysis."""
    
    # Base assessment data
    report = {
        "metadata": {
            "ticker": ticker,
            "report_date": datetime.now().isoformat(),
            "report_version": "2.0",
            "analysis_period": "1Y",
            "data_sources": ["yfinance", "news_api", "fraud_detection", "market_analysis"]
        },
        "executive_summary": generate_executive_summary(assessment),
        "company_profile": assessment.get("company", {}),
        "overall_score": assessment.get("score", {}),
        "detailed_analysis": assessment.get("engines", {}),
        "risk_breakdown": {
            "overall_risk_level": "LOW" if assessment.get("score", {}).get("true_scalability_score", 0) > 50 else "MEDIUM" if assessment.get("score", {}).get("true_scalability_score", 0) > 30 else "HIGH",
            "primary_risk_factors": [],
            "mitigation_factors": []
        },
        "comparative_analysis": {
            "sector_performance": "N/A",
            "industry_ranking": "N/A", 
            "market_cap_tier": get_market_cap_tier(assessment.get("company", {})),
            "growth_vs_peers": "N/A"
        },
        "actionable_insights": generate_actionable_insights(assessment),
        "data_quality": {
            "financial_data_completeness": "HIGH",
            "news_data_coverage": "MEDIUM",
            "market_data_quality": "HIGH",
            "peer_comparison_available": assessment.get("engines", {}).get("peer", {}).get("peer_count", 0) > 0
        }
    }
    
    # Add risk factors
    engines = assessment.get("engines", {})
    
    # Primary risk factors
    if engines.get("fraud", {}).get("fraud_probability", 0) > 0.15:
        report["risk_breakdown"]["primary_risk_factors"].append("Elevated fraud risk indicators")
    
    if engines.get("market", {}).get("market_confidence", 0) < 0.3:
        report["risk_breakdown"]["primary_risk_factors"].append("Weak market confidence")
    
    if engines.get("news", {}).get("news_risk", 0) > 0.5:
        report["risk_breakdown"]["primary_risk_factors"].append("Negative news sentiment")
    
    # Mitigation factors
    if engines.get("fraud", {}).get("fraud_probability", 0) < 0.1:
        report["risk_breakdown"]["mitigation_factors"].append("Low fraud probability")
    
    if engines.get("market", {}).get("market_confidence", 0) > 0.6:
        report["risk_breakdown"]["mitigation_factors"].append("Strong market confidence")
    
    if engines.get("peer", {}).get("peer_score", 0) > 0.6:
        report["risk_breakdown"]["mitigation_factors"].append("Outperforming peers")
    
    return report


def get_market_cap_tier(company_info: Dict[str, Any]) -> str:
    """Determine market cap tier from company info."""
    market_cap = company_info.get("market_cap", 0)
    
    if market_cap > 200_000_000_000:  # $200B+
        return "MEGA_CAP"
    elif market_cap > 10_000_000_000:  # $10B-$200B
        return "LARGE_CAP"
    elif market_cap > 2_000_000_000:  # $2B-$10B
        return "MID_CAP"
    elif market_cap > 300_000_000:  # $300M-$2B
        return "SMALL_CAP"
    else:
        return "MICRO_CAP"


def generate_actionable_insights(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Generate actionable insights and recommendations."""
    score = assessment.get("score", {})
    engines = assessment.get("engines", {})
    
    tss = score.get("true_scalability_score", 0)
    fraud_prob = engines.get("fraud", {}).get("fraud_probability", 0)
    market_conf = engines.get("market", {}).get("market_confidence", 0)
    
    insights = {
        "investment_recommendation": "HOLD",
        "risk_tolerance": "MODERATE",
        "time_horizon": "MEDIUM_TERM",
        "key_concerns": [],
        "opportunities": [],
        "monitoring_points": []
    }
    
    # Investment recommendation
    if tss > 60 and fraud_prob < 0.15:
        insights["investment_recommendation"] = "BUY"
    elif tss < 30 or fraud_prob > 0.25:
        insights["investment_recommendation"] = "SELL"
    
    # Risk tolerance
    if fraud_prob < 0.1 and market_conf > 0.6:
        insights["risk_tolerance"] = "LOW"
    elif fraud_prob > 0.2 or market_conf < 0.3:
        insights["risk_tolerance"] = "HIGH"
    
    # Key concerns
    if fraud_prob > 0.2:
        insights["key_concerns"].append("Elevated fraud risk requires due diligence")
    
    if market_conf < 0.3:
        insights["key_concerns"].append("Weak market sentiment may impact short-term performance")
    
    # Opportunities
    if tss > 50 and fraud_prob < 0.15:
        insights["opportunities"].append("Strong fundamentals suggest long-term growth potential")
    
    if market_conf > 0.6:
        insights["opportunities"].append("Positive market momentum may drive near-term gains")
    
    # Monitoring points
    insights["monitoring_points"] = [
        "Quarterly earnings consistency",
        "Regulatory developments",
        "Peer group performance",
        "Market sentiment trends",
        "News sentiment changes"
    ]
    
    return insights


def export_to_csv(assessment: Dict[str, Any], ticker: str, filename: Optional[str] = None) -> str:
    """Export assessment data to CSV format."""
    if filename is None:
        filename = f"reports/{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Flatten data for CSV
    csv_data = []
    
    # Basic info
    csv_data.append(["Metric", "Value", "Category"])
    csv_data.append(["Ticker", ticker, "Basic Info"])
    csv_data.append(["Report Date", datetime.now().strftime('%Y-%m-%d'), "Basic Info"])
    
    # Company info
    company = assessment.get("company", {})
    csv_data.append(["Company Name", company.get("name", ""), "Company Info"])
    csv_data.append(["Sector", company.get("sector", ""), "Company Info"])
    csv_data.append(["Industry", company.get("industry", ""), "Company Info"])
    csv_data.append(["Market Cap", company.get("market_cap", ""), "Company Info"])
    
    # Scores
    score = assessment.get("score", {})
    csv_data.append(["True Scalability Score", score.get("true_scalability_score", ""), "Scores"])
    csv_data.append(["Grade", score.get("grade", ""), "Scores"])
    
    # Engine results
    engines = assessment.get("engines", {})
    
    # Fraud
    fraud = engines.get("fraud", {})
    csv_data.append(["Fraud Probability", fraud.get("fraud_probability", ""), "Risk Analysis"])
    csv_data.append(["Fraud Risk Level", fraud.get("risk_level", ""), "Risk Analysis"])
    
    # Market
    market = engines.get("market", {})
    csv_data.append(["Market Confidence", market.get("market_confidence", ""), "Market Analysis"])
    csv_data.append(["Market Signal", market.get("market_signal", ""), "Market Analysis"])
    csv_data.append(["30D Momentum", market.get("momentum_30d", ""), "Market Analysis"])
    
    # News
    news = engines.get("news", {})
    csv_data.append(["News Risk", news.get("news_risk", ""), "Sentiment Analysis"])
    csv_data.append(["News Risk Level", news.get("risk_level", ""), "Sentiment Analysis"])
    csv_data.append(["Mean Sentiment", news.get("mean_sentiment", ""), "Sentiment Analysis"])
    
    # Peer
    peer = engines.get("peer", {})
    csv_data.append(["Peer Score", peer.get("peer_score", ""), "Peer Analysis"])
    csv_data.append(["Peer Signal", peer.get("signal", ""), "Peer Analysis"])
    
    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    return filename


def generate_html_report(assessment: Dict[str, Any], ticker: str) -> str:
    """Generate HTML report (if Jinja2 available)."""
    if not JINJA2_AVAILABLE:
        return "Jinja2 not available for HTML generation"
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corporate Intelligence Report - {{ ticker }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .score { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .grade { padding: 5px 10px; border-radius: 3px; color: white; }
        .grade-A { background: #27ae60; }
        .grade-B { background: #f39c12; }
        .grade-C { background: #e67e22; }
        .grade-D { background: #e74c3c; }
        .grade-F { background: #c0392b; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .risk-low { color: #27ae60; }
        .risk-medium { color: #f39c12; }
        .risk-high { color: #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f4f4f4; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Corporate Intelligence Report</h1>
        <h2>{{ company_name }} ({{ ticker }})</h2>
        <p>Generated: {{ report_date }}</p>
    </div>
    
    <div class="section">
        <h3>Overall Assessment</h3>
        <div class="score">{{ true_scalability_score }}/100</div>
        <div class="grade grade-{{ grade }}">{{ grade }}</div>
    </div>
    
    <div class="section">
        <h3>Company Profile</h3>
        <table>
            <tr><th>Attribute</th><th>Value</th></tr>
            <tr><td>Sector</td><td>{{ sector }}</td></tr>
            <tr><td>Industry</td><td>{{ industry }}</td></tr>
            <tr><td>Market Cap</td><td>${{ "{:,.0f}".format(market_cap) }}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Risk Analysis</h3>
        <table>
            <tr><th>Risk Factor</th><th>Value</th><th>Level</th></tr>
            <tr><td>Fraud Probability</td><td>{{ fraud_prob }}%</td><td class="risk-{{ fraud_class }}">{{ fraud_level }}</td></tr>
            <tr><td>Market Confidence</td><td>{{ market_conf }}%</td><td class="risk-{{ market_class }}">{{ market_signal }}</td></tr>
            <tr><td>News Risk</td><td>{{ news_risk }}%</td><td class="risk-{{ news_class }}">{{ news_level }}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Executive Summary</h3>
        <pre>{{ executive_summary }}</pre>
    </div>
</body>
</html>
    """
    
    template = Template(html_template)
    
    # Prepare template data
    company = assessment.get("company", {})
    score = assessment.get("score", {})
    engines = assessment.get("engines", {})
    
    html_content = template.render(
        ticker=ticker,
        company_name=company.get("name", ticker),
        report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        true_scalability_score=score.get("true_scalability_score", 0),
        grade=score.get("grade", "N/A"),
        sector=company.get("sector", ""),
        industry=company.get("industry", ""),
        market_cap=company.get("market_cap", 0),
        fraud_prob=engines.get("fraud", {}).get("fraud_probability", 0) * 100,
        fraud_level=engines.get("fraud", {}).get("risk_level", ""),
        fraud_class="low" if engines.get("fraud", {}).get("fraud_probability", 0) < 0.15 else "medium" if engines.get("fraud", {}).get("fraud_probability", 0) < 0.25 else "high",
        market_conf=engines.get("market", {}).get("market_confidence", 0) * 100,
        market_signal=engines.get("market", {}).get("market_signal", ""),
        market_class="low" if engines.get("market", {}).get("market_confidence", 0) > 0.6 else "medium" if engines.get("market", {}).get("market_confidence", 0) > 0.3 else "high",
        news_risk=engines.get("news", {}).get("news_risk", 0) * 100,
        news_level=engines.get("news", {}).get("risk_level", ""),
        news_class="low" if engines.get("news", {}).get("news_risk", 0) < 0.3 else "medium" if engines.get("news", {}).get("news_risk", 0) < 0.5 else "high",
        executive_summary=generate_executive_summary(assessment)
    )
    
    return html_content


def generate_comprehensive_report(assessment: Dict[str, Any], ticker: str, 
                                output_dir: str = "reports") -> Dict[str, str]:
    """
    Generate comprehensive reports in multiple formats.
    
    Parameters
    ----------
    assessment : Dict[str, Any]
        Full assessment results from the pipeline
    ticker : str
        Stock ticker symbol
    output_dir : str
        Directory to save reports
        
    Returns
    -------
    Dict[str, str]
        Dictionary with file paths for each generated report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{output_dir}/{ticker}_comprehensive_{timestamp}"
    
    generated_files = {}
    
    # 1. Detailed JSON report
    json_filename = f"{base_filename}.json"
    detailed_report = generate_detailed_json(assessment, ticker)
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)
    generated_files["json"] = json_filename
    
    # 2. Executive summary text
    summary_filename = f"{base_filename}_summary.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(generate_executive_summary(assessment))
    generated_files["summary"] = summary_filename
    
    # 3. CSV export
    csv_filename = export_to_csv(assessment, ticker, f"{base_filename}.csv")
    generated_files["csv"] = csv_filename
    
    # 4. HTML report (if Jinja2 available)
    if JINJA2_AVAILABLE:
        html_filename = f"{base_filename}.html"
        html_content = generate_html_report(assessment, ticker)
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        generated_files["html"] = html_filename
    
    # 5. PDF report (if WeasyPrint available)
    if JINJA2_AVAILABLE and WEASYPRINT_AVAILABLE:
        try:
            pdf_filename = f"{base_filename}.pdf"
            html_content = generate_html_report(assessment, ticker)
            HTML(string=html_content).write_pdf(pdf_filename)
            generated_files["pdf"] = pdf_filename
        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
    
    return generated_files


if __name__ == "__main__":
    # Test the report generator
    print("🧪 Testing Report Generator...")
    
    # Sample assessment data
    sample_assessment = {
        "ticker": "AAPL",
        "company": {
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000
        },
        "score": {
            "true_scalability_score": 45.21,
            "grade": "C"
        },
        "engines": {
            "fraud": {
                "fraud_probability": 0.136,
                "risk_level": "LOW"
            },
            "market": {
                "market_confidence": 0.143,
                "market_signal": "BEARISH",
                "momentum_30d": -0.1075
            },
            "news": {
                "news_risk": 0.46,
                "risk_level": "MEDIUM",
                "mean_sentiment": 0.08
            },
            "peer": {
                "peer_score": 0.417,
                "signal": "IN_LINE"
            }
        }
    }
    
    # Generate reports
    files = generate_comprehensive_report(sample_assessment, "AAPL")
    
    print("\n📄 Generated Reports:")
    for format_type, filepath in files.items():
        print(f"  {format_type.upper()}: {filepath}")
    
    print("\n✅ Report generation test completed!")
