"""
Gemini API Client Module

This module provides functionality to analyze earnings call transcripts
using the Gemini API, with proper rate limiting and error handling.
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GEMINI_API_KEY = 'AIzaSyCLCbdNsmtNl1IS9S95kPOtMphMESeVND8'
RATE_LIMIT = 60  # calls per minute

class RateLimiter:
    """Rate limiter for Gemini API calls."""
    
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

class GeminiClient:
    """Client for interacting with Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rate_limiter = RateLimiter()
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze an earnings call transcript.
        
        Args:
            transcript: The transcript text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Get summary
            summary = self.get_summary(transcript)
            if 'error' in summary:
                return summary
            
            # Get sentiment
            sentiment = self.get_sentiment(summary['summary'])
            if 'error' in sentiment:
                return {
                    'summary': summary['summary'],
                    'sentiment_error': sentiment['error']
                }
            
            # Get prediction
            prediction = self.predict_stock_movement(summary['summary'], sentiment['sentiment_score'])
            if 'error' in prediction:
                return {
                    'summary': summary['summary'],
                    'sentiment_score': sentiment['sentiment_score'],
                    'prediction_error': prediction['error']
                }
            
            return {
                'summary': summary['summary'],
                'sentiment_score': sentiment['sentiment_score'],
                'predicted_movement': prediction['movement']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return {'error': str(e)}
    
    def get_summary(self, transcript: str) -> Dict:
        """Generate a concise summary of the transcript."""
        try:
            prompt = f"""Please provide a concise one paragraph summary of this earnings call transcript:

{transcript}

Summary:"""
            
            response = self.model.generate_content(prompt)
            return {'summary': response.text.strip()}
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {'error': str(e)}
    
    def get_sentiment(self, summary: str) -> Dict:
        """Generate a sentiment score based on the summary."""
        try:
            prompt = f"""Based on this earnings call summary, provide a sentiment score between -1 and 1, where:
-1 is extremely negative
0 is neutral
1 is extremely positive

Summary: {summary}

Please provide your response in this exact format:
Sentiment Score: [number between -1 and 1]"""
            
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract sentiment score using regex
            import re
            match = re.search(r'Sentiment Score:\s*([-+]?\d*\.?\d+)', text)
            if not match:
                return {'error': 'Could not extract sentiment score from response'}
            
            score = float(match.group(1))
            if not -1 <= score <= 1:
                return {'error': f'Sentiment score {score} is out of range [-1, 1]'}
            
            return {'sentiment_score': score}
            
        except Exception as e:
            logger.error(f"Error generating sentiment: {str(e)}")
            return {'error': str(e)}
    
    def predict_stock_movement(self, summary: str, sentiment_score: float) -> Dict:
        """Predict short-term stock movement based on summary and sentiment."""
        try:
            prompt = f"""Based on this earnings call summary and sentiment score, predict the short-term stock movement.
Summary: {summary}
Sentiment Score: {sentiment_score}

Please predict if the stock will go UP, DOWN, or remain FLAT in the short term.
Provide your response in this exact format:
Prediction: [UP/DOWN/FLAT]"""
            
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract prediction
            import re
            match = re.search(r'Prediction:\s*(UP|DOWN|FLAT)', text, re.IGNORECASE)
            if not match:
                return {'error': 'Could not extract prediction from response'}
            
            movement = match.group(1).upper()
            if movement not in ['UP', 'DOWN', 'FLAT']:
                return {'error': f'Invalid prediction: {movement}'}
            
            return {'movement': movement.lower()}
            
        except Exception as e:
            logger.error(f"Error predicting movement: {str(e)}")
            return {'error': str(e)}
