"""
Stock Correlation Module

This module provides functionality to correlate earnings call sentiment
with stock price movements and build predictive models.
"""

from .price_fetcher import fetch_stock_prices
from .data_processor import merge_sentiment_prices
from .models import train_regression_model, train_classification_model
from .visualizer import (
    plot_sentiment_returns,
    plot_model_performance,
    plot_prediction_vs_actual
)

__all__ = [
    'fetch_stock_prices',
    'merge_sentiment_prices',
    'train_regression_model',
    'train_classification_model',
    'plot_sentiment_returns',
    'plot_model_performance',
    'plot_prediction_vs_actual'
] 