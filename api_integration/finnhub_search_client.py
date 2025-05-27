"""
Finnhub Search Client Module

This module provides functionality to search for companies using the Finnhub API,
with proper rate limiting and error handling.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
FINNHUB_API_KEY = 'd0nvndpr01qn5ghmb0d0d0nvndpr01qn5ghmb0dg'
RATE_LIMIT = 60  # calls per minute

class FinnhubSearchClient:
    """Client for searching companies using Finnhub API."""
    
    def __init__(self):
        """Initialize the Finnhub search client."""
        self.api_key = FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.last_call_time = datetime.now()
        self.calls_this_minute = 0
        self.rate_limit = RATE_LIMIT
        
    def _wait_for_rate_limit(self):
        """Wait if rate limit is reached."""
        now = datetime.now()
        if (now - self.last_call_time).seconds >= 60:
            self.calls_this_minute = 0
            self.last_call_time = now
        
        if self.calls_this_minute >= self.rate_limit:
            sleep_time = 60 - (now - self.last_call_time).seconds
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls_this_minute = 0
            self.last_call_time = datetime.now()
        
        self.calls_this_minute += 1
        
    def search_companies(self, query: str) -> List[Dict]:
        """
        Search for companies with rate limiting and error handling.
        
        Args:
            query: Search string (company name or ticker)
            
        Returns:
            List of company information dictionaries containing:
            - symbol: Stock ticker
            - description: Company name
            - type: Security type
            - primaryExchange: Primary exchange
        """
        if not query or len(query) < 2:
            return []
            
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('result', []):
                if 'description' in item and 'symbol' in item:
                    results.append({
                        'ticker': item['symbol'],
                        'name': item['description'],
                        'type': item.get('type', 'N/A'),
                        'exchange': item.get('primaryExchange', 'N/A')
                    })
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching companies: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in company search: {str(e)}")
            return [] 