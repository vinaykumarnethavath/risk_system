"""
news_api.py — Live News API Service
======================================
Integrates with real news APIs to fetch current headlines
for sentiment analysis instead of using sample headlines.

Supported APIs:
- NewsAPI.org (requires API key)
- RSS feeds (free)
- Alpha Vantage News API (requires API key)

Usage:
    from services.news_api import get_live_headlines
    headlines = get_live_headlines("AAPL")
"""

import requests
import feedparser
import re
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# For testing - you can add your keys here directly
NEWS_API_KEY = "9b93466a935a41efa65f00c38486c947"
ALPHA_VANTAGE_KEY = "1CH28YCTS7AO9DPQ"

# Try to load API keys from environment (overrides direct assignment)
# NEWS_API_KEY = os.getenv("NEWS_API_KEY") or NEWS_API_KEY
# ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY") or ALPHA_VANTAGE_KEY


def clean_headline(headline: str) -> str:
    """Clean and normalize headline text."""
    # Remove HTML tags
    headline = re.sub(r'<[^>]+>', '', headline)
    # Remove extra whitespace
    headline = re.sub(r'\s+', ' ', headline).strip()
    # Remove source prefixes like "[Source] " (only if it matches exactly)
    headline = re.sub(r'^\[[^\]]+\]\s*', '', headline)
    return headline


def get_newsapi_headlines(ticker: str, company_name: str = None, max_headlines: int = 10) -> List[str]:
    """
    Fetch headlines from NewsAPI.org (requires API key).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    company_name : str, optional
        Full company name for better search results
    max_headlines : int
        Maximum number of headlines to return
        
    Returns
    -------
    List[str]
        List of headline strings
    """
    if not NEWS_API_KEY:
        print("⚠️ NewsAPI key not found. Set NEWS_API_KEY environment variable.")
        return []
    
    try:
        headlines = []
        
        # First try company-specific search
        search_terms = []
        if company_name:
            search_terms.append(company_name.split()[0])  # First word of company name
        search_terms.append(ticker)
        
        print(f"  🔍 Searching for: {search_terms}")
        
        for query in search_terms:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "q": query,
                "language": "en",
                "pageSize": max_headlines,
                "apiKey": NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            print(f"  📡 Query '{query}' status: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            print(f"  📄 Found {len(articles)} articles for '{query}'")
            
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                
                full_headline = f"{title}. {description}" if description else title
                full_headline = clean_headline(full_headline)
                
                if len(full_headline) > 20 and full_headline not in headlines:
                    headlines.append(full_headline)
            
            if len(headlines) >= max_headlines:
                break
        
        # If no company-specific news, get general business news
        if not headlines:
            print("  🏢 No company news found, trying business category...")
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": "business",
                "language": "en",
                "pageSize": max_headlines,
                "apiKey": NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            print(f"  📡 Business category status: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            print(f"  📄 Found {len(articles)} business articles")
            
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                
                full_headline = f"{title}. {description}" if description else title
                full_headline = clean_headline(full_headline)
                
                if len(full_headline) > 20 and full_headline not in headlines:
                    headlines.append(full_headline)
        
        print(f"  ✅ Total headlines: {len(headlines)}")
        return headlines[:max_headlines]
        
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return []


def get_rss_headlines(ticker: str, company_name: str = None, max_headlines: int = 10) -> List[str]:
    """
    Fetch headlines from free RSS feeds (Yahoo Finance, etc.).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    company_name : str, optional
        Full company name
    max_headlines : int
        Maximum number of headlines
        
    Returns
    -------
    List[str]
        List of headline strings
    """
    headlines = []
    
    try:
        # Yahoo Finance RSS feed
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:max_headlines]:
            title = clean_headline(entry.get("title", ""))
            summary = clean_headline(entry.get("summary", ""))
            
            # Combine title and summary
            full_headline = f"{title}. {summary}" if summary else title
            
            if len(full_headline) > 20:
                headlines.append(full_headline)
        
        return headlines[:max_headlines]
        
    except Exception as e:
        print(f"❌ RSS feed error for {ticker}: {e}")
        return []


def get_alpha_vantage_headlines(ticker: str, max_headlines: int = 10) -> List[str]:
    """
    Fetch headlines from Alpha Vantage News API (requires API key).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    max_headlines : int
        Maximum number of headlines
        
    Returns
    -------
    List[str]
        List of headline strings
    """
    if not ALPHA_VANTAGE_KEY:
        print("⚠️ Alpha Vantage key not found. Set ALPHA_VANTAGE_KEY environment variable.")
        return []
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        headlines = []
        
        for article in data.get("feed", [])[:max_headlines]:
            title = clean_headline(article.get("title", ""))
            summary = clean_headline(article.get("summary", ""))
            
            full_headline = f"{title}. {summary}" if summary else title
            
            if len(full_headline) > 20:
                headlines.append(full_headline)
        
        return headlines[:max_headlines]
        
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        return []


def get_fallback_headlines(ticker: str) -> List[str]:
    """
    Fallback headlines when APIs fail (based on ticker patterns).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
        
    Returns
    -------
    List[str]
        Generic headlines
    """
    fallback_templates = [
        f"{ticker} reports quarterly earnings",
        f"Analysts maintain neutral outlook on {ticker}",
        f"{ticker} announces strategic partnership",
        f"Market volatility impacts {ticker} stock price",
        f"{ticker} faces regulatory scrutiny",
        f"Investors cautious about {ticker} prospects",
        f"{ticker} stock movement reflects broader market trends",
        f"Technical analysis suggests mixed signals for {ticker}",
    ]
    
    return fallback_templates[:6]


def get_live_headlines(ticker: str, company_name: str = None, max_headlines: int = 10) -> List[str]:
    """
    Get live headlines from multiple sources with fallback.
    
    Priority order:
    1. NewsAPI.org (if key available)
    2. Alpha Vantage (if key available) 
    3. RSS feeds (free)
    4. Fallback templates
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    company_name : str, optional
        Full company name
    max_headlines : int
        Maximum number of headlines to return
        
    Returns
    -------
    List[str]
        List of headline strings
    """
    print(f"📰 Fetching live news for {ticker}...")
    
    # Try NewsAPI first
    if NEWS_API_KEY:
        headlines = get_newsapi_headlines(ticker, company_name, max_headlines)
        if headlines:
            print(f"  ✓ NewsAPI: Found {len(headlines)} headlines")
            return headlines
    
    # Try Alpha Vantage second
    if ALPHA_VANTAGE_KEY:
        headlines = get_alpha_vantage_headlines(ticker, max_headlines)
        if headlines:
            print(f"  ✓ Alpha Vantage: Found {len(headlines)} headlines")
            return headlines
    
    # Try RSS feeds
    headlines = get_rss_headlines(ticker, company_name, max_headlines)
    if headlines:
        print(f"  ✓ RSS: Found {len(headlines)} headlines")
        return headlines
    
    # Use fallback
    print(f"  ⚠️ Using fallback headlines (no API keys available)")
    return get_fallback_headlines(ticker)


def test_news_sources():
    """Test all news sources with sample tickers."""
    test_tickers = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corporation"},
        {"ticker": "TSLA", "name": "Tesla, Inc."}
    ]
    
    print("🧪 Testing news sources...")
    print("=" * 50)
    
    for test in test_tickers:
        print(f"\n📊 Testing {test['ticker']} ({test['name']}):")
        headlines = get_live_headlines(test['ticker'], test['name'], max_headlines=3)
        
        for i, headline in enumerate(headlines, 1):
            print(f"  {i}. {headline}")


if __name__ == "__main__":
    test_news_sources()
