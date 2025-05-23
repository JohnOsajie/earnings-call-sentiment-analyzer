"""
Visualizer Module

This module provides functions for visualizing sentiment analysis
results and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_sentiment_returns(df: pd.DataFrame,
                         return_col: str = 'one_day_return',
                         save_path: Optional[str] = None) -> None:
    """
    Create scatter plot of sentiment vs returns.
    
    Args:
        df (pd.DataFrame): Dataframe with sentiment and return data
        return_col (str): Column name for returns
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    sns.scatterplot(data=df, x='compound', y=return_col)
    
    # Add regression line
    sns.regplot(data=df, x='compound', y=return_col, scatter=False, color='red')
    
    # Customize plot
    plt.title('Sentiment vs. Stock Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_return_distribution(df: pd.DataFrame,
                           return_col: str = 'one_day_return',
                           save_path: Optional[str] = None) -> None:
    """
    Create distribution plot of returns.
    
    Args:
        df (pd.DataFrame): Dataframe with return data
        return_col (str): Column name for returns
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create distribution plot
    sns.histplot(data=df, x=return_col, bins=30)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Customize plot
    plt.title('Distribution of Stock Returns')
    plt.xlabel('Return')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_performance(metrics: Dict,
                         model_type: str = 'regression',
                         save_path: Optional[str] = None) -> None:
    """
    Create bar plot of model performance metrics.
    
    Args:
        metrics (Dict): Dictionary of performance metrics
        model_type (str): Type of model ('regression' or 'classification')
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if model_type == 'regression':
        # Plot regression metrics
        metrics_to_plot = {
            'MSE': metrics['mse'],
            'R²': metrics['r2']
        }
    else:
        # Plot classification metrics
        metrics_to_plot = {
            'Accuracy': metrics['accuracy']
        }
    
    # Create bar plot
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    
    # Customize plot
    plt.title(f'{model_type.title()} Model Performance')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metrics_to_plot.values()):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_prediction_vs_actual(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model_type: str = 'regression',
                            save_path: Optional[str] = None) -> None:
    """
    Create plot comparing predicted vs actual values.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        model_type (str): Type of model ('regression' or 'classification')
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if model_type == 'regression':
        # Create scatter plot for regression
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Predicted vs Actual Returns')
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
    else:
        # Create confusion matrix for classification
        cm = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
    
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 