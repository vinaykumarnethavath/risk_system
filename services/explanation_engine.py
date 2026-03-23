"""
explanation_engine.py — GenAI Explanation Engine
================================================
Generates analyst-style narrative reports from the scoring
results using structured prompts sent to OpenAI's GPT models.

Falls back to a template-based explanation if no API key is set.
"""

import os
import json

from config import OPENAI_MODEL


# ─── Prompt Template ─────────────────────────────────────────────────


ANALYST_PROMPT = """You are a senior financial analyst at a top-tier investment bank.
Generate a concise, professional corporate intelligence report based on the following data.

Company: {company_name} ({ticker})
Sector: {sector} | Industry: {industry}

═══ SCORING SUMMARY ═══
True Scalability Score: {score}/100 (Grade: {grade})

═══ SIGNAL BREAKDOWN ═══
• Fraud Probability: {fraud_prob:.1%} ({fraud_level})
• Adjusted Growth: {adjusted_growth:.2%} (Trust Factor: {trust_factor:.2%})
• Market Confidence: {market_confidence:.1%} ({market_signal})
  - 30d Momentum: {momentum_30d:.2%}
  - Annual Volatility: {annual_volatility:.2%}
  - Max Drawdown: {max_drawdown:.2%}
• News Risk: {news_risk:.1%} ({news_level})
  - Mean Sentiment: {mean_sentiment:.3f}
• Peer Position: {peer_score:.1%} ({peer_signal})

═══ INSTRUCTIONS ═══
Write a 3-paragraph analyst report covering:
1. Executive overview with the key score and what it means for investors
2. Risk analysis highlighting the most significant flags (fraud, market, news)
3. Forward-looking recommendation with specific action items

Use professional financial language. Be direct and actionable.
Do NOT use bullet points — write in flowing paragraphs.
"""


# ─── GenAI Explanation (OpenAI) ──────────────────────────────────────


def explain_with_genai(assessment: dict) -> str:
    """
    Generate an AI-powered analyst explanation using OpenAI.

    Parameters
    ----------
    assessment : dict
        Full assessment from fass_core.generate_full_assessment().

    Returns
    -------
    str
        Generated analyst narrative report.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ OPENAI_API_KEY not set — using template-based explanation.")
        return explain_with_template(assessment)

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        prompt = _build_prompt(assessment)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior financial analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        print("  ⚠ openai package not installed — using template-based explanation.")
        return explain_with_template(assessment)
    except Exception as e:
        print(f"  ⚠ GenAI error: {e} — using template-based explanation.")
        return explain_with_template(assessment)


# ─── Template-Based Fallback ─────────────────────────────────────────


def explain_with_template(assessment: dict) -> str:
    """
    Generate a template-based explanation when GenAI is unavailable.

    Parameters
    ----------
    assessment : dict
        Full assessment from fass_core.generate_full_assessment().

    Returns
    -------
    str
        Structured analyst-style report.
    """
    score = assessment["score"]
    engines = assessment["engines"]
    info = assessment["company"]
    ticker = assessment["ticker"]

    tss = score["true_scalability_score"]
    grade = score["grade"]
    name = info.get("name", ticker)

    fraud = engines["fraud"]
    market = engines["market"]
    news = engines["news"]
    peer = engines["peer"]
    adj = engines["adjustment"]

    # Paragraph 1: Executive Overview
    if tss >= 70:
        outlook = "strong"
        recommendation = "presents a compelling case for investment consideration"
    elif tss >= 50:
        outlook = "moderate"
        recommendation = "warrants cautious optimism with selective positioning"
    elif tss >= 30:
        outlook = "mixed"
        recommendation = "reflects neutral signals with moderate risk indicators"
    else:
        outlook = "concerning"
        recommendation = "signals elevated risk that demands careful due diligence"

    para1 = (
        f"{name} ({ticker}) receives a True Scalability Score of {tss}/100 "
        f"(Grade: {grade}), reflecting a {outlook} corporate intelligence profile. "
        f"The company, operating in the {info.get('sector', 'N/A')} sector, "
        f"{recommendation}. Our multi-signal analysis integrates forensic fraud "
        f"detection, market momentum, news sentiment, and peer benchmarking to "
        f"deliver this assessment."
    )

    # Paragraph 2: Risk Analysis
    risk_flags = []
    if fraud.get("fraud_probability", 0) >= 0.3:
        risk_flags.append(
            f"elevated fraud probability of {fraud['fraud_probability']:.1%}"
        )
    if news.get("news_risk", 0.5) >= 0.55:
        risk_flags.append(
            f"negative news sentiment (risk: {news['news_risk']:.1%})"
        )
    if market.get("market_confidence", 0.5) < 0.4:
        risk_flags.append(
            f"weak market confidence at {market['market_confidence']:.1%}"
        )

    if risk_flags:
        flags_text = ", ".join(risk_flags)
        para2 = (
            f"Our risk analysis identifies the following concerns: {flags_text}. "
            f"The fraud-adjusted growth rate stands at "
            f"{adj.get('adjusted_growth', 0):.2%}, reflecting a trust factor of "
            f"{adj.get('trust_factor', 1):.2%} applied to raw financial figures. "
            f"Market analysis shows a {market.get('market_signal', 'NEUTRAL')} "
            f"signal with {market.get('momentum_30d', 0):.2%} thirty-day momentum "
            f"and a maximum drawdown of {market.get('max_drawdown', 0):.2%}."
        )
    else:
        para2 = (
            f"The risk profile appears well-contained. Fraud probability is "
            f"measured at {fraud.get('fraud_probability', 0):.1%} "
            f"({fraud.get('risk_level', 'N/A')}), news sentiment is "
            f"{'positive' if news.get('news_risk', 0.5) < 0.45 else 'neutral'}, "
            f"and market confidence stands at "
            f"{market.get('market_confidence', 0.5):.1%} "
            f"({market.get('market_signal', 'NEUTRAL')}). "
            f"The fraud-adjusted growth rate of "
            f"{adj.get('adjusted_growth', 0):.2%} suggests financials are "
            f"largely trustworthy."
        )

    # Paragraph 3: Recommendation
    peer_text = (
        f"Relative to peers, the company is "
        f"{peer.get('signal', 'N/A').lower().replace('_', ' ')} "
        f"with a peer score of {peer.get('peer_score', 0.5):.1%}."
    )

    if tss >= 70:
        action = (
            "We recommend a Buy/Overweight stance based on strong signal convergence."
        )
    elif tss >= 50:
        action = (
            "We recommend a Hold position with close monitoring."
        )
    elif tss >= 30:
        action = (
            "We maintain a Neutral stance, watching for better entry points or signal resolution."
        )
    else:
        action = (
            "We recommend reducing exposure due to elevated risk factors."
        )

    para3 = f"{peer_text} {action}"

    return f"{para1}\n\n{para2}\n\n{para3}"


# ─── Prompt Builder ──────────────────────────────────────────────────


def _build_prompt(assessment: dict) -> str:
    """Build the structured prompt for GenAI from assessment data."""
    score = assessment["score"]
    engines = assessment["engines"]
    info = assessment["company"]

    return ANALYST_PROMPT.format(
        company_name=info.get("name", assessment["ticker"]),
        ticker=assessment["ticker"],
        sector=info.get("sector", "N/A"),
        industry=info.get("industry", "N/A"),
        score=score["true_scalability_score"],
        grade=score["grade"],
        fraud_prob=engines["fraud"].get("fraud_probability", 0),
        fraud_level=engines["fraud"].get("risk_level", "N/A"),
        adjusted_growth=engines["adjustment"].get("adjusted_growth", 0),
        trust_factor=engines["adjustment"].get("trust_factor", 1),
        market_confidence=engines["market"].get("market_confidence", 0.5),
        market_signal=engines["market"].get("market_signal", "N/A"),
        momentum_30d=engines["market"].get("momentum_30d", 0),
        annual_volatility=engines["market"].get("annual_volatility", 0),
        max_drawdown=engines["market"].get("max_drawdown", 0),
        news_risk=engines["news"].get("news_risk", 0.5),
        news_level=engines["news"].get("risk_level", "N/A"),
        mean_sentiment=engines["news"].get("mean_sentiment", 0),
        peer_score=engines["peer"].get("peer_score", 0.5),
        peer_signal=engines["peer"].get("signal", "N/A"),
    )


# ─── Convenience Wrapper ─────────────────────────────────────────────


def generate_report(assessment: dict, use_genai: bool = True) -> str:
    """
    Generate the final analyst report.

    Parameters
    ----------
    assessment : dict
        Full assessment from fass_core.generate_full_assessment().
    use_genai : bool
        If True, attempt GenAI; otherwise use template.

    Returns
    -------
    str
        Complete analyst narrative report.
    """
    if use_genai:
        return explain_with_genai(assessment)
    else:
        return explain_with_template(assessment)


if __name__ == "__main__":
    # Demo with mock assessment
    mock_assessment = {
        "ticker": "AAPL",
        "company": {
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        "score": {
            "true_scalability_score": 72.5,
            "grade": "B+",
        },
        "engines": {
            "fraud": {"fraud_probability": 0.12, "risk_level": "LOW"},
            "adjustment": {"adjusted_growth": 0.13, "trust_factor": 0.88},
            "market": {
                "market_confidence": 0.68,
                "market_signal": "BULLISH",
                "momentum_30d": 0.05,
                "annual_volatility": 0.22,
                "max_drawdown": -0.12,
            },
            "news": {
                "news_risk": 0.35,
                "risk_level": "LOW",
                "mean_sentiment": 0.3,
            },
            "peer": {"peer_score": 0.72, "signal": "OUTPERFORMING"},
        },
    }

    report = generate_report(mock_assessment, use_genai=False)
    print("\n" + "=" * 60)
    print("  ANALYST REPORT")
    print("=" * 60)
    print(report)
