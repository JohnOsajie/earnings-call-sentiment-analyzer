"""
Sentiment Processor

This module provides functionality to process and analyze sentiment data
from various sources.
"""

from typing import Dict, List
import pandas as pd
from datetime import datetime

def process_sentiment_data(sentiment_data: Dict) -> Dict:
    """
    Process and analyze sentiment data.
    
    Args:
        sentiment_data (Dict): Raw sentiment data from API
        
    Returns:
        Dict: Processed sentiment data with additional analysis
    """
    # Create a copy to avoid modifying original
    processed = sentiment_data.copy()
    
    # Add sentiment category
    score = processed['sentiment_score']
    if score > 0.2:
        category = "Positive"
    elif score < -0.2:
        category = "Negative"
    else:
        category = "Neutral"
    processed['sentiment_category'] = category
    
    # Process articles if available
    if 'articles' in processed:
        processed['articles'] = _process_articles(processed['articles'])
        
        # Calculate article statistics
        if processed['articles']:
            processed['article_stats'] = {
                'total_articles': len(processed['articles']),
                'avg_sentiment': sum(a['sentiment'] for a in processed['articles']) / len(processed['articles']),
                'positive_articles': sum(1 for a in processed['articles'] if a['sentiment'] > 0),
                'negative_articles': sum(1 for a in processed['articles'] if a['sentiment'] < 0),
                'neutral_articles': sum(1 for a in processed['articles'] if -0.1 <= a['sentiment'] <= 0.1)
            }
    
    return processed

def _process_articles(articles: List[Dict]) -> List[Dict]:
    """
    Process and clean article data.
    
    Args:
        articles (List[Dict]): List of article dictionaries
        
    Returns:
        List[Dict]: Processed article data
    """
    processed = []
    for article in articles:
        # Clean and process each article
        processed_article = {
            'title': article.get('title', '').strip(),
            'url': article.get('url', ''),
            'summary': article.get('summary', '').strip(),
            'sentiment': float(article.get('sentiment', 0)),
            'published': _parse_date(article.get('published', '')),
            'category': _categorize_sentiment(float(article.get('sentiment', 0)))
        }
        processed.append(processed_article)
    
    # Sort by date (most recent first)
    processed.sort(key=lambda x: x['published'], reverse=True)
    
    return processed

def _parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object."""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return datetime.now()

def _categorize_sentiment(score: float) -> str:
    """Categorize sentiment score."""
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral" 