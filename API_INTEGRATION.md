# Earnings Call Sentiment API Integration

## Overview
This feature integrates with Finnhub.io's API to fetch sentiment analysis for earnings calls. The integration provides an additional data source for sentiment analysis, complementing our existing NLP-based analysis.

## API Details
- **Provider**: Finnhub.io
- **API Key**: Securely stored in environment variables
- **Endpoint**: `/news-sentiment`
- **Rate Limits**: Free tier (60 calls/minute)
- **Data Type**: News sentiment (fallback for earnings-specific sentiment)

## Feature Requirements

### 1. API Integration Module
- Create `api_integration/` directory
- Implement `finnhub_client.py` for API interactions
- Implement `sentiment_processor.py` for data processing
- Add proper error handling and rate limiting

### 2. Core Functions

#### `get_sentiment(ticker: str, date: datetime, api_key: str) -> Dict`
- Parameters:
  - `ticker`: Stock symbol (e.g., "AAPL")
  - `date`: Target date for earnings call
  - `api_key`: Finnhub API key
- Returns:
  - Dictionary containing:
    - `sentiment_score`: Overall sentiment (-1 to 1)
    - `buzz_score`: News volume score
    - `summary`: Brief summary of sentiment
    - `articles`: List of relevant articles
    - `nearest_date`: Actual date of data (if different from input)

### 3. Security Measures
- Store API key in environment variables
- Implement API key validation
- Add request rate limiting
- Handle API errors gracefully
- Log API usage for monitoring

### 4. Streamlit Integration
- Add new tab "API Sentiment"
- Input fields:
  - Stock ticker
  - Date picker
  - Optional: Date range
- Display:
  - Sentiment metrics
  - News summary
  - Relevant articles
  - Comparison with our analysis

### 5. Error Handling
- API connection errors
- Rate limit exceeded
- Invalid ticker
- No data available
- Date out of range

### 6. Data Processing
- Clean and normalize API responses
- Convert dates to consistent format
- Filter relevant news articles
- Calculate aggregate sentiment

## Implementation Steps

1. **Setup**
   - Create API integration module
   - Set up environment variables
   - Install required packages

2. **Core Implementation**
   - Implement API client
   - Add sentiment processing
   - Create utility functions

3. **Streamlit Integration**
   - Add new tab
   - Create input forms
   - Implement visualization

4. **Testing**
   - Unit tests for API client
   - Integration tests
   - Error handling tests

## Dependencies
```python
finnhub-python>=2.4.18
python-dotenv>=0.19.0
requests>=2.31.0
```

## Usage Example
```python
from api_integration.finnhub_client import get_sentiment

# Get sentiment for AAPL on a specific date
sentiment_data = get_sentiment(
    ticker="AAPL",
    date="2024-02-01",
    api_key="your_api_key"
)

# Display results
print(f"Sentiment Score: {sentiment_data['sentiment_score']}")
print(f"Buzz Score: {sentiment_data['buzz_score']}")
print(f"Summary: {sentiment_data['summary']}")
```

## Notes
- Finnhub.io provides general news sentiment, not earnings-specific
- Consider implementing caching to reduce API calls
- Monitor API usage to stay within free tier limits
- Consider adding more data sources in the future

## Future Enhancements
1. Add multiple API sources
2. Implement sentiment comparison
3. Add historical sentiment tracking
4. Create sentiment trends visualization
5. Add custom date range analysis 