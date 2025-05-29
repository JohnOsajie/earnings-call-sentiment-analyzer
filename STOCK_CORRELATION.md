# Stock Price Correlation Feature

## Overview
This feature extends the Earnings Call Sentiment Analyzer to correlate sentiment scores with stock price movements, enabling users to analyze the relationship between earnings call sentiment and subsequent stock performance.

## Objectives
- Correlate earnings call sentiment with stock price movements
- Build predictive models for stock returns based on sentiment
- Visualize the relationship between sentiment and price movements

## Data Requirements

### Input Data
- Earnings call transcript sentiment scores
- Company ticker symbols
- Earnings call dates

### Price Data (via yfinance)
- Closing prices:
  - 1 day before earnings call
  - 1 day after earnings call
  - 7 days after earnings call

## Process Flow

1. **Data Collection**
   - Fetch historical stock prices using yfinance
   - Handle missing data (market holidays, etc.)
   - Calculate price returns:
     - 1-day return = (price_day_after - price_before) / price_before
     - 1-week return = (price_week_after - price_before) / price_before

2. **Data Integration**
   - Merge sentiment scores with price return data
   - Create unified dataframe for analysis

3. **Modeling**
   - Linear Regression:
     - Predict return percentage from sentiment score
     - Evaluate model performance
   - Binary Classification:
     - Predict price movement direction (up/down)
     - Calculate accuracy and other metrics

4. **Visualization**
   - Scatter plots of sentiment vs. returns
   - Distribution of returns by sentiment
   - Model performance metrics

## 🛠️ Technical Implementation

### Dependencies
```python
yfinance>=0.2.36
scikit-learn>=1.4.0
seaborn>=0.13.0
```

### File Structure
```
earnings-call-sentiment-analyzer/
├── stock_correlation/
│   ├── __init__.py
│   ├── price_fetcher.py    # yfinance integration
│   ├── data_processor.py   # Data cleaning and merging
│   ├── models.py          # ML models
│   └── visualizer.py      # Plotting functions
└── tests/
    └── test_stock_correlation.py
```

### Key Functions

#### Price Fetcher
```python
def fetch_stock_prices(ticker: str, date: datetime) -> Dict:
    """
    Fetch stock prices around earnings call date.
    Returns dict with prices and calculated returns.
    """
```

#### Data Processor
```python
def merge_sentiment_prices(sentiment_df: pd.DataFrame, 
                         price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment and price data into unified dataframe.
    """
```

#### Models
```python
def train_regression_model(X: np.array, y: np.array) -> sklearn.Model:
    """
    Train linear regression model for return prediction.
    """

def train_classification_model(X: np.array, y: np.array) -> sklearn.Model:
    """
    Train binary classifier for price movement prediction.
    """
```

## Output

### Data
- Merged dataframe with:
  - Sentiment scores
  - Price returns
  - Predicted values
  - Model performance metrics

### Visualizations
- Sentiment vs. Return scatter plots
- Return distribution by sentiment
- Model performance charts

### Models
- Trained regression model
- Trained classification model
- Model evaluation metrics

## Testing
- Unit tests for data processing
- Integration tests for yfinance API
- Model performance validation
- Edge case handling (market holidays, missing data)

## Notes
- Handle market holidays and missing data appropriately
- Consider timezone differences for international stocks
- Account for stock splits and dividends
- Implement proper error handling for API calls
- Cache price data to minimize API calls

## Future Enhancements
- Add more sophisticated models (Random Forest, XGBoost)
- Include additional features (market conditions, sector performance)
- Implement backtesting capabilities
- Add real-time prediction functionality
- Create interactive visualizations 
