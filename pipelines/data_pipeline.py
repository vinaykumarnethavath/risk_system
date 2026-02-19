"""
data_pipeline.py — Financial and stock data ingestion pipeline.
===============================================================
Provides functions to fetch quarterly financials, stock price
history, balance sheet data, and key financial ratios via yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf


# ─── Core Data Fetchers ─────────────────────────────────────────────


def get_financials(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly financial statements for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL', 'TSLA').

    Returns
    -------
    pd.DataFrame
        Transposed quarterly financials with dates as rows.
    """
    tk = yf.Ticker(ticker)
    financials = tk.quarterly_financials.T
    financials.index.name = "date"
    return financials


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock price data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    period : str
        Lookback period (default '1y'). Options: '1mo', '3mo',
        '6mo', '1y', '2y', '5y', 'max'.

    Returns
    -------
    pd.DataFrame
        OHLCV price data with datetime index.
    """
    data = yf.download(ticker, period=period, progress=False)
    return data


def get_balance_sheet(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly balance sheet data.

    Returns
    -------
    pd.DataFrame
        Transposed quarterly balance sheet.
    """
    tk = yf.Ticker(ticker)
    bs = tk.quarterly_balance_sheet.T
    bs.index.name = "date"
    return bs


def get_cash_flow(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly cash flow statement.

    Returns
    -------
    pd.DataFrame
        Transposed quarterly cash flow data.
    """
    tk = yf.Ticker(ticker)
    cf = tk.quarterly_cashflow.T
    cf.index.name = "date"
    return cf


def get_company_info(ticker: str) -> dict:
    """
    Fetch basic company information (sector, industry, market cap, etc.).

    Returns
    -------
    dict
        Company metadata.
    """
    tk = yf.Ticker(ticker)
    info = tk.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "country": info.get("country", "Unknown"),
        "website": info.get("website", ""),
        "employees": info.get("fullTimeEmployees", 0),
    }


# ─── Derived Financial Ratios ───────────────────────────────────────


def compute_financial_ratios(ticker: str) -> pd.DataFrame:
    """
    Compute key financial health ratios from quarterly financials
    and balance sheet.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: revenue_growth, profit_margin,
        debt_to_equity, current_ratio.
    """
    try:
        fin = get_financials(ticker)
        bs = get_balance_sheet(ticker)

        ratios = pd.DataFrame(index=fin.index)

        # Revenue growth (quarter-over-quarter)
        if "Total Revenue" in fin.columns:
            rev = fin["Total Revenue"]
            ratios["revenue_growth"] = rev.pct_change(-1)  # vs previous quarter

        # Profit margin
        if "Net Income" in fin.columns and "Total Revenue" in fin.columns:
            ratios["profit_margin"] = fin["Net Income"] / fin["Total Revenue"].replace(0, np.nan)

        # Debt to equity
        if "Total Debt" in bs.columns and "Stockholders Equity" in bs.columns:
            equity = bs["Stockholders Equity"].reindex(ratios.index, method="nearest")
            debt = bs["Total Debt"].reindex(ratios.index, method="nearest")
            ratios["debt_to_equity"] = debt / equity.replace(0, np.nan)

        ratios = ratios.replace([np.inf, -np.inf], np.nan)
        return ratios

    except Exception as e:
        print(f"Warning: Could not compute ratios for {ticker}: {e}")
        return pd.DataFrame()


# ─── Convenience Aggregator ─────────────────────────────────────────


def fetch_all_data(ticker: str, period: str = "1y") -> dict:
    """
    Fetch all available data for a ticker in one call.

    Returns
    -------
    dict
        Keys: 'info', 'financials', 'balance_sheet', 'cash_flow',
              'stock', 'ratios'.
    """
    print(f"[Pipeline] Fetching data for {ticker}...")

    info = get_company_info(ticker)
    print(f"  ✓ Company info: {info['name']} ({info['sector']})")

    financials = get_financials(ticker)
    print(f"  ✓ Quarterly financials: {len(financials)} quarters")

    balance_sheet = get_balance_sheet(ticker)
    print(f"  ✓ Balance sheet: {len(balance_sheet)} quarters")

    cash_flow = get_cash_flow(ticker)
    print(f"  ✓ Cash flow: {len(cash_flow)} quarters")

    stock = get_stock_data(ticker, period=period)
    print(f"  ✓ Stock data: {len(stock)} trading days")

    ratios = compute_financial_ratios(ticker)
    print(f"  ✓ Financial ratios computed")

    return {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "stock": stock,
        "ratios": ratios,
    }


if __name__ == "__main__":
    # Quick test
    data = fetch_all_data("AAPL")
    print("\n--- Financials (last 2 quarters) ---")
    print(data["financials"].head(2))
    print("\n--- Stock Data (last 5 days) ---")
    print(data["stock"].tail(5))
