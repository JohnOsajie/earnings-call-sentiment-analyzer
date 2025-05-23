"""
Data Processor Module

This module handles data cleaning and merging of sentiment and price data.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare sentiment data.
    
    Args:
        df (pd.DataFrame): Raw sentiment data
        
    Returns:
        pd.DataFrame: Cleaned sentiment data
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['ticker', 'date', 'compound']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker', 'date'])
    
    return df

def merge_sentiment_prices(sentiment_df: pd.DataFrame, 
                         price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment and price data into unified dataframe.
    
    Args:
        sentiment_df (pd.DataFrame): Sentiment data
        price_df (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    # Clean sentiment data
    sentiment_df = clean_sentiment_data(sentiment_df)
    
    # Convert price_df date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(price_df['date']):
        price_df['date'] = pd.to_datetime(price_df['date'])
    
    # Merge dataframes
    merged_df = pd.merge(
        sentiment_df,
        price_df,
        on=['ticker', 'date'],
        how='inner'
    )
    
    # Add binary movement column
    merged_df['price_movement'] = np.where(
        merged_df['one_day_return'] > 0,
        1,  # Up
        0   # Down
    )
    
    return merged_df

def prepare_model_data(df: pd.DataFrame, 
                      target_col: str = 'one_day_return') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        df (pd.DataFrame): Merged dataframe
        target_col (str): Column to use as target variable
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (target)
    """
    # Select features and target
    X = df[['compound']].values
    y = df[target_col].values
    
    return X, y 