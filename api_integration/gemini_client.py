import google.generativeai as genai
from typing import Dict, Any
import re  # added for sentiment extraction

class GeminiClient:
    def __init__(self, api_key: str):
        """Initialize Gemini client with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def get_summary(self, transcript: str) -> Dict[str, Any]:
        """
        Generate a concise summary of the earnings call transcript.
        Returns a summary in 2-3 sentences.
        """
        prompt = f"""
        Write a concise 2-3 sentence summary of this earnings call transcript:

        {transcript}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            return {
                'summary': text
            }
        except Exception as e:
            return {
                'error': f"Failed to generate summary: {str(e)}"
            }

    def get_sentiment(self, summary: str) -> Dict[str, Any]:
        """
        Generate a sentiment score based on the summary.
        Returns a sentiment score from -1 to 1.
        """
        prompt = f"""
        Given this summary of an earnings call, provide a sentiment score as a float between -1 (very negative) and 1 (very positive).
        ONLY return the number in this format exactly:
        
        SENTIMENT: [float between -1 and 1]
        Do NOT include any explanation.

        Summary:
        {summary}
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Use regex to extract sentiment score
            match = re.search(r"SENTIMENT:\s*(-?\d+(?:\.\d+)?)", text)
            if not match:
                return {
                    'error': f"Could not extract a valid sentiment score from: {text}"
                }

            try:
                sentiment_score = float(match.group(1))
                if not -1 <= sentiment_score <= 1:
                    return {
                        'error': f"Sentiment score {sentiment_score} is outside valid range [-1, 1]"
                    }
                return {
                    'sentiment_score': sentiment_score
                }
            except ValueError:
                return {
                    'error': f"Regex matched value, but failed to convert to float: {match.group(1)}"
                }

        except Exception as e:
            return {
                'error': f"Failed to generate sentiment score: {str(e)}"
            }

    def predict_stock_movement(self, summary: str, sentiment_score: float) -> Dict[str, Any]:
        """
        Predict short-term stock movement based on summary and sentiment score.
        Returns one of: 'up', 'down', or 'flat'.
        """
        prompt = f"""
        Based on the following earnings call summary and sentiment score, predict the short-term stock movement (within 1-5 days).
        Output ONLY one of: up, down, or flat.

        Summary: {summary}
        Sentiment Score: {sentiment_score}

        Prediction (only output the word):
        """

        try:
            response = self.model.generate_content(prompt)
            prediction = response.text.strip().lower()

            # Basic validation
            if prediction not in ['up', 'down', 'flat']:
                return {
                    'error': f"Invalid prediction response: {prediction}"
                }

            return {
                'predicted_movement': prediction
            }

        except Exception as e:
            return {
                'error': f"Failed to predict stock movement: {str(e)}"
            }

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze earnings call transcript using Gemini.
        Returns summary, sentiment score, and predicted short-term stock movement.
        """
        # Get summary
        summary_result = self.get_summary(transcript)
        if 'error' in summary_result:
            return summary_result
        summary = summary_result['summary']

        # Get sentiment
        sentiment_result = self.get_sentiment(summary)
        if 'error' in sentiment_result:
            return {
                'summary': summary,
                'sentiment_error': sentiment_result['error']
            }
        sentiment_score = sentiment_result['sentiment_score']

        # Predict movement
        movement_result = self.predict_stock_movement(summary, sentiment_score)
        result = {
            'summary': summary,
            'sentiment_score': sentiment_score
        }

        if 'error' in movement_result:
            result['prediction_error'] = movement_result['error']
        else:
            result['predicted_movement'] = movement_result['predicted_movement']

        return result
