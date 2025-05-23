import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with VADER."""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict: Dictionary containing sentiment analysis results
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Analyze each paragraph
        results = []
        for i, para in enumerate(paragraphs):
            scores = self.sia.polarity_scores(para)
            results.append({
                'paragraph': i + 1,
                'text': para,
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg'],
                'compound': scores['compound']
            })
        
        # Calculate overall sentiment
        df = pd.DataFrame(results)
        overall = {
            'compound': df['compound'].mean(),
            'positive': df['positive'].mean(),
            'neutral': df['neutral'].mean(),
            'negative': df['negative'].mean(),
            'paragraphs': results
        }
        
        return overall
    
    def display_results(self, results: Dict) -> None:
        """
        Display the sentiment analysis results.
        
        Args:
            results (Dict): The sentiment analysis results
        """
        print("\nSentiment Analysis Results:")
        print("-" * 30)
        print(f"Overall Compound Sentiment: {results['compound']:.2f}")
        print(f"Positive: {results['positive']*100:.0f}%")
        print(f"Neutral: {results['neutral']*100:.0f}%")
        print(f"Negative: {results['negative']*100:.0f}%")
    
    def save_results(self, results: Dict) -> None:
        """
        Save the results to CSV and create visualization.
        
        Args:
            results (Dict): The sentiment analysis results
        """
        # Save detailed results to CSV
        df = pd.DataFrame(results['paragraphs'])
        df.to_csv('sentiment_scores.csv', index=False)
        
        # Create and save visualization
        self._create_visualization(results['paragraphs'])
    
    def _create_visualization(self, paragraphs: List[Dict]) -> None:
        """
        Create and save a visualization of the sentiment analysis.
        
        Args:
            paragraphs (List[Dict]): List of paragraph sentiment results
        """
        df = pd.DataFrame(paragraphs)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['paragraph'], df['compound'], label='Compound')
        plt.plot(df['paragraph'], df['positive'], label='Positive')
        plt.plot(df['paragraph'], df['negative'], label='Negative')
        
        plt.title('Sentiment Analysis Over Time')
        plt.xlabel('Paragraph Number')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('sentiment_plot.png')
        plt.close() 