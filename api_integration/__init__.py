"""
API Integration Module

This module provides functionality to fetch and process sentiment data
from external APIs like Finnhub.io.
"""

from .finnhub_client import get_sentiment
from .sentiment_processor import process_sentiment_data

__all__ = [
    'get_sentiment',
    'process_sentiment_data'
] 