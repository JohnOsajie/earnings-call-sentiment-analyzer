"""
Finnhub API Client Module

This module provides functionality to fetch stock price data and news
from the Finnhub API, with proper rate limiting and error handling.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import finnhub
import logging
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variable
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY environment variable is not set")

# Get rate limit from environment variable, default to 60 if not set
RATE_LIMIT = int(os.getenv('FINNHUB_RATE_LIMIT', 60))

class RateLimiter:
    """Rate limiter for Finnhub API calls."""
    
    def __init__(self):
        self.calls_per_minute = RATE_LIMIT
        self.calls = 0
        self.last_reset = datetime.now()
    
    def wait_if_needed(self):
        """Wait if rate limit is reached."""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.calls = 0
            self.last_reset = now
        
        if self.calls >= self.calls_per_minute:
            sleep_time = 60 - (now - self.last_reset).seconds
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
            self.calls = 0
            self.last_reset = datetime.now()
        
        self.calls += 1

# Initialize rate limiter
rate_limiter = RateLimiter()

def get_stock_price_on_date(symbol: str, date_str: str) -> dict:
    """
    Fetch stock price data for a specific date and the following 6 days using yfinance.
    If the market was closed on the given date, returns data for the nearest trading day.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        date_str: Target date in 'YYYY-MM-DD' format
    
    Returns:
        Dictionary containing:
        - date: The actual date used (may be different from input if fallback occurred)
        - open: Opening price
        - high: Highest price
        - low: Lowest price
        - close: Closing price
        - volume: Trading volume
        - is_fallback: Boolean indicating if a fallback date was used
        - following_days: List of dictionaries containing price data for T+1 to T+6
    """
    try:
        logger.info(f"Fetching stock price data for {symbol} on {date_str}")
        
        # Convert date string to datetime
        target_date = pd.to_datetime(date_str)
        
        # Try up to 5 previous days to find a trading day
        for days_back in range(5):
            current_date = target_date - pd.Timedelta(days=days_back)
            logger.info(f"Trying date: {current_date.strftime('%Y-%m-%d')}")
            
            # Get data for the current date and following 6 days
            stock = yf.Ticker(symbol)
            
            # Calculate date range (current date to T+6)
            end_date = current_date + pd.Timedelta(days=7)  # +7 to include T+6
            
            # Get historical data
            hist = stock.history(
                start=current_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            logger.info(f"Received historical data with shape: {hist.shape if not hist.empty else 'empty'}")
            
            if not hist.empty:
                # Get the specific day's data
                day_data = hist[hist.index.date == current_date.date()]
                logger.info(f"Found {len(day_data)} rows for target date")
                
                if not day_data.empty:
                    # Process following days data
                    following_days = []
                    for date, row in hist.iterrows():
                        if date.date() > current_date.date():  # Only include days after the target date
                            following_days.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'close': float(row['Close']),
                                'volume': int(row['Volume'])
                            })
                    
                    result = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'open': float(day_data['Open'].iloc[0]),
                        'high': float(day_data['High'].iloc[0]),
                        'low': float(day_data['Low'].iloc[0]),
                        'close': float(day_data['Close'].iloc[0]),
                        'volume': int(day_data['Volume'].iloc[0]),
                        'is_fallback': days_back > 0,
                        'following_days': following_days
                    }
                    
                    logger.info("Successfully fetched and processed price data")
                    return result
        
        # If no data found after trying 5 days
        logger.warning("No trading data found within 5 days of the target date")
        return {
            'date': date_str,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None,
            'is_fallback': False,
            'following_days': [],
            'error': 'No trading data found within 5 days of the target date'
        }
        
    except Exception as e:
        logger.error(f"Error fetching stock price data: {str(e)}")
        return {
            'date': date_str,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None,
            'is_fallback': False,
            'following_days': [],
            'error': str(e)
        }

def get_stock_prices(ticker: str, date: datetime) -> Dict:
    """
    Fetch stock prices for T to T+6 using yfinance.
    
    Args:
        ticker: Stock symbol
        date: Target date (T)
    
    Returns:
        Dictionary containing price data for 7 days
    """
    try:
        # Calculate date range (T to T+6)
        start_date = date
        end_date = date + pd.Timedelta(days=6)
        
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Add 1 day to include T+6
        )
        
        if hist.empty:
            logger.warning(f"No data found for {ticker} from {start_date} to {end_date}")
            return None
            
        # Process the data
        prices = {}
        for i, (date, row) in enumerate(hist.iterrows()):
            day_key = f'T+{i}' if i > 0 else 'T'
            prices[day_key] = {
                'date': date.strftime('%Y-%m-%d'),
                'price': row['Close'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row['Volume'],
                'change': ((row['Close'] - row['Open']) / row['Open']) * 100
            }
            
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching stock prices: {str(e)}")
        return None

def get_sentiment(ticker: str, date: datetime) -> Dict:
    """
    Fetch news articles for a stock on a specific date.
    
    Args:
        ticker: Stock symbol
        date: Target date
    
    Returns:
        Dictionary containing articles data
    """
    try:
        # Check if date is too old (Finnhub only provides last 7 days of news)
        today = datetime.now().date()
        max_start_date = today - pd.Timedelta(days=7)
        
        if date.date() < max_start_date:
            return {
                'summary': f'Note: Finnhub only provides news data for the last 7 days. Your requested date ({date.date()}) is too old.',
                'articles': []
            }
        
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        
        # Calculate date range (±3 days) using pd.Timedelta
        start_date = (date - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        end_date = (date + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Apply rate limiting
        rate_limiter.wait_if_needed()
        
        # Fetch news
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        
        if not news:
            return {
                'summary': 'No news data available for the selected date range',
                'articles': []
            }
        
        # Process articles (limit to 10)
        processed_articles = _process_articles(news[:10])
        
        return {
            'summary': f"Found {len(processed_articles)} recent articles about {ticker}",
            'articles': processed_articles
        }
        
    except finnhub.exceptions.FinnhubAPIException as e:
        logger.error(f"Finnhub API error: {str(e)}")
        return {
            'summary': f'Finnhub API error: {str(e)}',
            'articles': []
        }
    except Exception as e:
        logger.error(f"Error fetching news data: {str(e)}")
        return {
            'summary': f'Error fetching data: {str(e)}',
            'articles': []
        }

def _process_articles(articles: List[Dict]) -> List[Dict]:
    """Process and clean article data."""
    processed = []
    for article in articles:
        processed.append({
            'title': article.get('headline', ''),
            'summary': article.get('summary', ''),
            'url': article.get('url', ''),
            'source': article.get('source', ''),
            'published': article.get('datetime', '')
        })
    return sorted(processed, key=lambda x: x['published'], reverse=True) 