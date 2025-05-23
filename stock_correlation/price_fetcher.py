"""
Price Fetcher Module

This module handles fetching and processing stock price data using yfinance.
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_nearest_earnings_date(ticker: str, target_date: datetime) -> Tuple[datetime, bool]:
    """
    Find the nearest earnings call date for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        target_date (datetime): Target date to find nearest earnings call
        
    Returns:
        Tuple[datetime, bool]: (nearest date, is_exact_match)
    """
    try:
        # Get earnings dates
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates
        
        if earnings_dates is None or earnings_dates.empty:
            logger.warning(f"No earnings dates found for {ticker}")
            return target_date, False
            
        # Convert index to datetime if needed
        if not isinstance(earnings_dates.index, pd.DatetimeIndex):
            earnings_dates.index = pd.to_datetime(earnings_dates.index)
            
        # Find nearest date
        earnings_dates = earnings_dates.sort_index()
        nearest_date = min(earnings_dates.index, key=lambda x: abs((x - target_date).days))
        
        # Check if it's an exact match
        is_exact_match = (nearest_date.date() == target_date.date())
        
        return nearest_date, is_exact_match
        
    except Exception as e:
        logger.error(f"Error finding nearest earnings date for {ticker}: {str(e)}")
        return target_date, False

def fetch_stock_prices(ticker: str, date: datetime) -> Dict:
    """
    Fetch stock prices around earnings call date.
    
    Args:
        ticker (str): Stock ticker symbol
        date (datetime): Target earnings call date
        
    Returns:
        Dict: Dictionary containing prices and calculated returns
    """
    try:
        # Find nearest earnings date
        nearest_date, is_exact_match = find_nearest_earnings_date(ticker, date)
        
        # Calculate date range (1 day before to 7 days after)
        start_date = nearest_date - timedelta(days=1)
        end_date = nearest_date + timedelta(days=7)
        
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No data found for {ticker} around {nearest_date}")
            return None
            
        # Get required prices
        price_before = hist.iloc[0]['Close']  # 1 day before
        price_after = hist.iloc[-1]['Close']  # Last available price
        
        # Calculate returns
        one_day_return = (price_after - price_before) / price_before
        
        # Get 7-day return if available
        if len(hist) >= 7:
            price_week_after = hist.iloc[6]['Close']
            one_week_return = (price_week_after - price_before) / price_before
        else:
            one_week_return = None
            
        return {
            'ticker': ticker,
            'date': nearest_date,
            'price_before': price_before,
            'price_after': price_after,
            'one_day_return': one_day_return,
            'one_week_return': one_week_return,
            'is_exact_match': is_exact_match
        }
        
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {str(e)}")
        return None

def fetch_batch_prices(tickers: list, dates: list) -> pd.DataFrame:
    """
    Fetch prices for multiple tickers and dates.
    
    Args:
        tickers (list): List of ticker symbols
        dates (list): List of dates
        
    Returns:
        pd.DataFrame: DataFrame with price data for all tickers
    """
    results = []
    
    for ticker, date in zip(tickers, dates):
        result = fetch_stock_prices(ticker, date)
        if result:
            results.append(result)
            
    return pd.DataFrame(results) 