"""
Models Module

This module provides functionality for training and evaluating
regression and classification models.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_regression_model(X: np.ndarray, 
                         y: np.ndarray,
                         test_size: float = 0.2) -> Tuple[LinearRegression, Dict]:
    """
    Train linear regression model for return prediction.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        test_size (float): Proportion of data to use for testing
        
    Returns:
        Tuple[LinearRegression, Dict]: Trained model and performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'coefficient': model.coef_[0],
        'intercept': model.intercept_
    }
    
    return model, metrics

def train_classification_model(X: np.ndarray,
                             y: np.ndarray,
                             test_size: float = 0.2) -> Tuple[LogisticRegression, Dict]:
    """
    Train binary classifier for price movement prediction.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values (0 for down, 1 for up)
        test_size (float): Proportion of data to use for testing
        
    Returns:
        Tuple[LogisticRegression, Dict]: Trained model and performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'coefficient': model.coef_[0][0],
        'intercept': model.intercept_[0]
    }
    
    return model, metrics

def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Evaluate model performance on new data.
    
    Args:
        model: Trained model (regression or classification)
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        
    Returns:
        Dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics based on model type
    if isinstance(model, LinearRegression):
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
    else:  # Classification model
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        }
    
    return metrics 