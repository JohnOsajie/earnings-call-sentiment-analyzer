"""
Finnhub API Client

This module provides functionality to interact with the Finnhub.io API
for fetching news data and calculating sentiment.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import finnhub
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_KEY = "d0nvndpr01qn5ghmb0d0d0nvndpr01qn5ghmb0dg"  # Your provided API key
RATE_LIMIT = 60  # calls per minute
RATE_WINDOW = 60  # seconds

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls = self.calls[1:]
        
        self.calls.append(now)

def get_sentiment(ticker: str, date: datetime, api_key: str = API_KEY) -> Dict:
    """
    Fetch news data for a given ticker and calculate sentiment.
    
    Args:
        ticker (str): Stock symbol (e.g., "AAPL")
        date (datetime): Target date for earnings call
        api_key (str): Finnhub API key
        
    Returns:
        Dict: Dictionary containing sentiment data
        
    Raises:
        ValueError: If ticker is invalid
        ConnectionError: If API request fails
        Exception: For other errors
    """
    try:
        # Initialize rate limiter
        limiter = RateLimiter(RATE_LIMIT)
        
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=api_key)
        
        # Calculate date range (1 day before and after)
        start_date = (date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Wait for rate limit if needed
        limiter.wait_if_needed()
        
        # Fetch company news
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        
        if not news:
            logger.warning(f"No news found for {ticker} around {date}")
            return {
                'sentiment_score': 0,
                'buzz_score': 0,
                'summary': f"No news available for {ticker} near {date.strftime('%Y-%m-%d')}",
                'articles': [],
                'nearest_date': date.strftime('%Y-%m-%d')
            }
        
        # Process articles and calculate sentiment
        processed_articles = _process_articles(news)
        sentiment_score = _calculate_sentiment(processed_articles)
        buzz_score = min(len(processed_articles) / 10, 1.0)  # Normalize to 0-1 range
        
        # Get the most recent article date
        nearest_date = date
        if processed_articles:
            try:
                article_dates = []
                for article in processed_articles:
                    if 'datetime' in article:
                        article_dates.append(datetime.fromtimestamp(article['datetime']))
                if article_dates:
                    nearest_date = max(article_dates)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing article dates: {str(e)}")
        
        # Generate summary
        summary = _generate_summary(sentiment_score, buzz_score, len(processed_articles))
        
        return {
            'sentiment_score': sentiment_score,
            'buzz_score': buzz_score,
            'summary': summary,
            'articles': processed_articles,
            'nearest_date': nearest_date.strftime('%Y-%m-%d')
        }
        
    except finnhub.exceptions.FinnhubAPIException as e:
        logger.error(f"Finnhub API error: {str(e)}")
        raise ConnectionError(f"Failed to fetch data from Finnhub: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        raise

def _calculate_sentiment(articles: List[Dict]) -> float:
    """Calculate sentiment score from articles."""
    if not articles:
        return 0.0
    
    # Simple sentiment calculation based on category
    sentiment_map = {
        'positive': 1.0,
        'neutral': 0.0,
        'negative': -1.0
    }
    
    total_sentiment = sum(sentiment_map.get(article.get('category', 'neutral'), 0) 
                         for article in articles)
    return total_sentiment / len(articles)

def _generate_summary(sentiment_score: float, buzz_score: float, article_count: int) -> str:
    """Generate a summary of the sentiment data."""
    if article_count == 0:
        return "No news data available"
    
    if sentiment_score > 0.2:
        sentiment_text = "positive"
    elif sentiment_score < -0.2:
        sentiment_text = "negative"
    else:
        sentiment_text = "neutral"
    
    return (f"Overall sentiment is {sentiment_text} (score: {sentiment_score:.2f}) "
            f"with {buzz_score:.1f} buzz score from {article_count} articles.")

def _process_articles(articles: List[Dict]) -> List[Dict]:
    """Process and clean article data."""
    processed = []
    for article in articles:
        try:
            # Convert datetime to timestamp if it's a string
            datetime_str = article.get('datetime', '')
            if isinstance(datetime_str, str):
                try:
                    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    timestamp = dt.timestamp()
                except ValueError:
                    timestamp = time.time()
            else:
                timestamp = datetime_str
            
            processed.append({
                'title': article.get('headline', ''),
                'url': article.get('url', ''),
                'summary': article.get('summary', ''),
                'category': article.get('category', 'neutral'),
                'datetime': timestamp,
                'published': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            logger.warning(f"Error processing article: {str(e)}")
            continue
    
    return processed 