"""
Earnings Call Sentiment Analyzer

This Streamlit app analyzes earnings call transcripts and their market impact
using sentiment analysis and stock price data.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from api_integration.gemini_client import GeminiClient

# Import custom modules
# from transcript_fetcher import fetch_transcript # We will primarily use pasted text, so commenting this out for now
from api_integration.finnhub_client import get_sentiment, get_stock_prices, get_stock_price_on_date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize Gemini client
GEMINI_API_KEY = "AIzaSyCxSDVjgljIVoPTIxUQV9AVhGTo4iak7m8"
gemini_client = GeminiClient(GEMINI_API_KEY)

# Commenting out earnings date fetching as manual input is default
# def get_earnings_dates(ticker: str) -> List[datetime]:
#     """
#     Get list of recent earnings dates for a stock.
    
#     Args:
#         ticker: Stock symbol
    
#     Returns:
#         List of earnings dates
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         earnings_dates = stock.earnings_dates
#         if earnings_dates is not None and not earnings_dates.empty:
#             # Ensure dates are sorted in descending order (most recent first)
#             return sorted(earnings_dates.index.tolist(), reverse=True)
#         return []
#     except Exception as e:
#         logger.error(f"Error fetching earnings dates: {str(e)}")
#         return []

# Commenting out nearest earnings date finding as manual input is default
# def find_nearest_earnings_date(ticker: str, target_date: datetime) -> Optional[datetime]:
#     """
#     Find the nearest PAST earnings date to the target date.
    
#     Args:
#         ticker: Stock symbol
#         target_date: Target date
    
#     Returns:
#         Nearest past earnings date or None if not found
#     """
#     earnings_dates = get_earnings_dates(ticker)
#     if not earnings_dates:
#         return None
    
#     # Convert target_date to datetime.date for comparison if it's a datetime object
#     if isinstance(target_date, datetime):
#         target_date_obj = target_date.date()
#     else:
#         target_date_obj = target_date # Assume it's already a date object
        
#     # Filter for earnings dates on or before the target date
#     past_earnings_dates = [date for date in earnings_dates if date.date() <= target_date_obj]

#     if not past_earnings_dates:
#         logger.warning(f"No past earnings dates found for {ticker} on or before {target_date_obj}")
#         return None

#     # Find the nearest past date
#     nearest_past_date = min(past_earnings_dates, key=lambda x: abs(x.date() - target_date_obj))
#     return nearest_past_date

def analyze_transcript_sentiment(transcript_text: str) -> List[Dict]:
    """
    Analyze sentiment of the transcript text by splitting into paragraphs.
    
    Args:
        transcript_text: Full transcript text as a string
    
    Returns:
        List of dictionaries with sentiment scores for each paragraph
    """
    # Split the transcript into paragraphs based on double newlines
    paragraphs = [p.strip() for p in transcript_text.split('\n\n') if p.strip()]
    
    results = []
    for i, para in enumerate(paragraphs):
        scores = sia.polarity_scores(para)
        results.append({
            'paragraph': para,
            'paragraph_number': i + 1,
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu']
        })
    return results

def generate_sentiment_plot(sentiment_data: List[Dict]) -> go.Figure:
    """
    Generate a plot of sentiment scores over paragraph number.
    
    Args:
        sentiment_data: List of sentiment analysis results
    
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame(sentiment_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['paragraph_number'],
        y=df['compound_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Sentiment Analysis Per Paragraph',
        xaxis_title='Paragraph Number',
        yaxis_title='Sentiment Score',
        hovermode='x unified'
    )
    
    return fig

def generate_sentiment_price_correlation_plot(avg_sentiment: float, price_data: Dict) -> go.Figure:
    """
    Generate a plot showing average sentiment vs. stock price movement.
    
    Args:
        avg_sentiment: Average compound sentiment score from the transcript.
        price_data: Dictionary of price data (T, T+1, T+7).
        
    Returns:
        Plotly figure object.
    """
    # We need the actual dates for the x-axis for a meaningful time series plot
    # Based on get_stock_prices, the keys are strings 'T', 'T+1', 'T+7'. Let's use these as labels.

    time_labels = ['T', 'T+1', 'T+7']
    valid_dates = []
    valid_prices = []
    valid_price_changes = []

    for label in time_labels:
        if label in price_data and price_data[label] and price_data[label].get('price') is not None and price_data[label].get('change') is not None:
            valid_dates.append(label)
            valid_prices.append(price_data[label]['price'])
            valid_price_changes.append(price_data[label]['change'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add average sentiment as a horizontal line
    if valid_dates:
        fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=[avg_sentiment] * len(valid_dates),
                mode='lines',
                name='Average Sentiment',
                line=dict(color='red', dash='dash')
            ),
            secondary_y=False # Sentiment on primary y-axis
        )

    # Add stock price movement
    if valid_dates:
        fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=valid_prices,
                mode='lines+markers',
                name='Stock Price',
                line=dict(color='blue')
            ),
            secondary_y=True # Price on secondary y-axis
        )
        
        # Add price change annotations
        for i, change in enumerate(valid_price_changes):
            # Only add annotation if both price and change are valid
            if i < len(valid_prices):
                fig.add_annotation(
                    x=valid_dates[i],
                    y=valid_prices[i],
                    text=f'{change:.1f}%',
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
                )


    fig.update_layout(
        title='Average Sentiment vs. Stock Price Movement',
        xaxis_title='Time Period Relative to Earnings Call',
        yaxis_title='Average Sentiment Score',
        yaxis2_title='Stock Price ($)',
        hovermode='x unified'
    )

    # Ensure y-axis for sentiment covers the possible range [-1, 1]
    fig.update_yaxes(range=[-1, 1], secondary_y=False)

    return fig

def generate_market_impact_plot(
    predicted_movement: str,
    sentiment_score: float,
    price_data: Dict[str, Any]
) -> go.Figure:
    """
    Generate plot comparing predicted vs actual stock movement.
    
    Args:
        predicted_movement: 'up', 'down', or 'flat'
        sentiment_score: Float between -1 and 1
        price_data: Dictionary with dates and prices
    
    Returns:
        Plotly figure object
    """
    # Extract dates and prices
    dates = []
    prices = []
    for key in sorted(price_data.keys(), key=lambda x: int(x.split('+')[1]) if '+' in x else 0):
        if price_data[key]:
            dates.append(price_data[key]['date'])
            prices.append(price_data[key]['price'])

    if not dates or not prices:
        raise ValueError("No valid price data available")

    # Calculate actual movement
    start_price = prices[0]
    end_price = prices[-1]
    price_change = ((end_price - start_price) / start_price) * 100
    
    # Determine actual movement
    if abs(price_change) < 1:
        actual_movement = 'flat'
    else:
        actual_movement = 'up' if price_change > 0 else 'down'
    
    # Check if prediction was correct
    correct = predicted_movement == actual_movement
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Stock Price',
        line=dict(
            color='green' if correct else 'red',
            width=2
        ),
        marker=dict(size=8)
    ))
    
    # Add prediction annotation at start
    fig.add_annotation(
        x=dates[0],
        y=prices[0],
        text=f"Predicted: {predicted_movement.upper()}<br>Sentiment: {sentiment_score:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    # Add outcome annotation at end
    fig.add_annotation(
        x=dates[-1],
        y=prices[-1],
        text=f"{'✅ Correct' if correct else '❌ Incorrect'}<br>Change: {price_change:+.1f}%",
        showarrow=False,
        font=dict(
            color='green' if correct else 'red',
            size=14
        ),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title="Market Impact: Prediction vs. Actual",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        template="plotly_white",
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title("📈 Earnings Call Sentiment Analyzer")
    
    # Input section
    st.subheader("Enter Earnings Call Details")
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL").upper()
    with col2:
        # Default to datetime.date for consistency with yfinance potentially
        date = st.date_input("Earnings Date", datetime.now().date())
    
    transcript_text = st.text_area("Paste Earnings Call Transcript Here", height=400)
    
    # Initialize state variables for results if they don't exist
    if 'sentiment_results' not in st.session_state:
        st.session_state.sentiment_results = None
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    if 'news_sentiment' not in st.session_state:
        st.session_state.news_sentiment = None
    if 'detailed_price_data' not in st.session_state:
        st.session_state.detailed_price_data = None
    
    # Buttons for actions
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] {
            gap: 0;
            justify-content: space-between;
        }
        .stButton {
            width: 100%;
            margin: 0 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])  # Four equal width columns
    with col1:
        analyze_button = st.button("Analyze Sentiment", use_container_width=True)
    with col2:
        fetch_market_button = st.button("Show Market Impact", use_container_width=True)
    with col3:
        fetch_news_button = st.button("News Articles", use_container_width=True)
    with col4:
        fetch_detailed_button = st.button("Fetch Price Data", use_container_width=True)
        
    # --- Logic for button clicks ---
    
    if analyze_button:
        if not transcript_text:
            st.warning("Please paste the earnings call transcript before analyzing.")
        else:
            with st.spinner("Analyzing transcript..."):
                result = gemini_client.analyze_transcript(transcript_text)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Always show the summary if available
                    st.subheader("Transcript Summary")
                    st.write(result['summary'])
                    
                    # Show sentiment score if available, otherwise show error
                    st.subheader("Sentiment Analysis")
                    if 'sentiment_score' in result:
                        sentiment_score = result['sentiment_score']
                        # Store in session state
                        st.session_state.sentiment_score = sentiment_score
                        # Color code based on sentiment
                        if sentiment_score > 0:
                            st.success(f"Sentiment Score: {sentiment_score:.2f}")
                        elif sentiment_score < 0:
                            st.error(f"Sentiment Score: {sentiment_score:.2f}")
                        else:
                            st.info(f"Sentiment Score: {sentiment_score:.2f}")
                    elif 'sentiment_error' in result:
                        st.warning(f"Could not generate sentiment score: {result['sentiment_error']}")

                    # Show predicted movement if available
                    st.subheader("Price Movement Prediction")
                    if 'predicted_movement' in result:
                        movement = result['predicted_movement']
                        # Store in session state
                        st.session_state.predicted_movement = movement
                        if movement == 'up':
                            st.success("Predicted Movement: 📈 Up")
                        elif movement == 'down':
                            st.error("Predicted Movement: 📉 Down")
                        else:
                            st.info("Predicted Movement: ↔️ Flat")
                    elif 'prediction_error' in result:
                        st.warning(f"Could not generate price prediction: {result['prediction_error']}")

    if fetch_market_button:
        if not ticker or not date:
             st.warning("Please enter both Stock Ticker and Earnings Date to fetch market data.")
        else:
            with st.spinner("Fetching market data..."):
                try:
                    # Use the provided date for fetching price
                    target_fetch_date = datetime.combine(date, datetime.min.time())
                    
                    # Fetch price data
                    price_data = get_stock_prices(ticker, target_fetch_date)
                    if price_data:
                        st.session_state.price_data = price_data
                        st.success("Market data fetching complete!")
                    else:
                        st.warning("No market data available for the selected date.")
                        st.session_state.price_data = None
                except Exception as e:
                    st.error(f"An error occurred while fetching market data: {str(e)}")
                    logger.error(f"Market data fetching error: {str(e)}")
                    st.session_state.price_data = None

    if fetch_news_button:
        if not ticker or not date:
            st.warning("Please enter both Stock Ticker and Earnings Date to fetch news articles.")
        else:
            with st.spinner("Fetching news articles..."):
                try:
                    # Use the provided date for fetching news
                    target_fetch_date = datetime.combine(date, datetime.min.time())
                    
                    # Fetch news sentiment
                    logger.info(f"Fetching news sentiment for {ticker} near {target_fetch_date.strftime('%Y-%m-%d')}")
                    news_sentiment = get_sentiment(ticker, target_fetch_date)
                    st.session_state.news_sentiment = news_sentiment
                    
                    if st.session_state.news_sentiment:
                        st.success("News articles fetching complete!")
                    else:
                        st.warning("No articles were retrieved.")
                except Exception as e:
                    st.error(f"An error occurred while fetching news articles: {str(e)}")
                    logger.error(f"News articles fetching error: {str(e)}")
                    st.session_state.news_sentiment = None

    if fetch_detailed_button:
        if not ticker or not date:
            st.warning("Please enter both Stock Ticker and Earnings Date to fetch detailed price data.")
        else:
            with st.spinner("Fetching detailed price data..."):
                try:
                    # Convert date to string format
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Fetch detailed price data
                    detailed_price_data = get_stock_price_on_date(ticker, date_str)
                    st.session_state.detailed_price_data = detailed_price_data
                    
                    if detailed_price_data.get('error'):
                        st.error(f"Error fetching detailed price data: {detailed_price_data['error']}")
                    else:
                        st.success("Detailed price data fetched successfully!")
                        if detailed_price_data['is_fallback']:
                            st.info(f"Note: Using data from {detailed_price_data['date']} as market was closed on {date_str}")
                except Exception as e:
                    st.error(f"An error occurred while fetching detailed price data: {str(e)}")
                    logger.error(f"Detailed price data fetching error: {str(e)}")
                    st.session_state.detailed_price_data = None

    # --- Display Results ---

    # Only show tabs if some results are available
    if st.session_state.sentiment_results or st.session_state.price_data or st.session_state.news_sentiment or st.session_state.detailed_price_data:
        st.header("Analysis Results")
        
        # Create tabs dynamically based on available data
        tab_titles = []
        tab_contents = []

        if st.session_state.sentiment_results:
            tab_titles.append("Sentiment Analysis")
            tab_contents.append("sentiment")
        
        if st.session_state.price_data:
            tab_titles.append("Market Impact")
            tab_contents.append("market")

        if st.session_state.news_sentiment:
            tab_titles.append("News Context")
            tab_contents.append("news")
            
        if st.session_state.detailed_price_data:
            tab_titles.append("Detailed Price Data")
            tab_contents.append("detailed_price")

        # Create tabs
        if tab_titles:
            tabs = st.tabs(tab_titles)
            
            for i, tab_content_key in enumerate(tab_contents):
                with tabs[i]:
                    if tab_content_key == "sentiment":
                        st.subheader("Transcript Sentiment")
                        st.plotly_chart(generate_sentiment_plot(st.session_state.sentiment_results))
                        
                        # Display sentiment metrics
                        avg_sentiment = sum(d['compound_score'] for d in st.session_state.sentiment_results) / len(st.session_state.sentiment_results)
                        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
                        
                        # Display transcript with sentiment
                        st.subheader("Transcript with Sentiment")
                        for item in st.session_state.sentiment_results:
                            with st.expander(f"Paragraph {item['paragraph_number']} (Score: {item['compound_score']:.2f})"):
                                st.write(item['paragraph'])

                    elif tab_content_key == "market" and st.session_state.price_data:
                        st.subheader("Stock Price Movement")
                        
                        try:
                            # Get prediction and sentiment from session state
                            if 'predicted_movement' in st.session_state and 'sentiment_score' in st.session_state:
                                # Display the market impact plot
                                st.plotly_chart(
                                    generate_market_impact_plot(
                                        st.session_state.predicted_movement,
                                        st.session_state.sentiment_score,
                                        st.session_state.price_data
                                    ),
                                    use_container_width=True
                                )
                            else:
                                st.warning("Please analyze the transcript first to get predictions.")
                        except Exception as e:
                            st.warning(f"Could not display Market Impact data. Error: {str(e)}")
                            logger.error(f"Market Impact display error: {str(e)}")

                    elif tab_content_key == "news" and st.session_state.news_sentiment:
                        st.subheader("Recent News Articles")
                        if 'summary' in st.session_state.news_sentiment:
                            st.write(st.session_state.news_sentiment['summary'])
                        
                        # Display news articles
                        if 'articles' in st.session_state.news_sentiment and st.session_state.news_sentiment['articles']:
                            for article in st.session_state.news_sentiment['articles']:
                                # Add checks for article keys
                                title = article.get('title', 'No Title')
                                summary = article.get('summary', 'No summary available.')
                                source = article.get('source', 'Unknown Source')
                                published = article.get('published', 'Unknown Date')
                                url = article.get('url', '#')

                                with st.expander(title):
                                    st.write(summary)
                                    st.write(f"Source: {source}")
                                    st.write(f"Published: {published}")
                                    if url != '#':
                                        st.link_button("Read More", url)
                        else:
                            st.info("No related articles found.")

                    elif tab_content_key == "detailed_price" and st.session_state.detailed_price_data:
                        st.subheader("Detailed Stock Price Data")
                        
                        price_data = st.session_state.detailed_price_data
                        logger.info(f"Price data received: {price_data}")
                        
                        # Display date information
                        st.write(f"**Date:** {price_data['date']}")
                        if price_data['is_fallback']:
                            st.info("This is data from the closest trading day as the market was closed on the requested date.")
                        
                        # Create a candlestick chart for the selected date and following days
                        if all(price_data.get(key) is not None for key in ['open', 'high', 'low', 'close']):
                            try:
                                # Prepare data for the chart
                                dates = [price_data['date']]
                                opens = [price_data['open']]
                                highs = [price_data['high']]
                                lows = [price_data['low']]
                                closes = [price_data['close']]
                                
                                # Add following days data
                                for day in price_data.get('following_days', []):
                                    dates.append(day['date'])
                                    opens.append(day['open'])
                                    highs.append(day['high'])
                                    lows.append(day['low'])
                                    closes.append(day['close'])
                                
                                # Create the candlestick chart
                                fig = go.Figure(data=[go.Candlestick(
                                    x=dates,
                                    open=opens,
                                    high=highs,
                                    low=lows,
                                    close=closes
                                )])
                                
                                # Add volume bars
                                volumes = [price_data['volume']] + [day['volume'] for day in price_data.get('following_days', [])]
                                fig.add_trace(go.Bar(
                                    x=dates,
                                    y=volumes,
                                    name='Volume',
                                    marker_color='rgba(0,0,255,0.3)',
                                    yaxis='y2'
                                ))
                                
                                fig.update_layout(
                                    title=f"{ticker} Price Data from {price_data['date']} to {dates[-1]}",
                                    yaxis_title="Price ($)",
                                    yaxis2=dict(
                                        title="Volume",
                                        overlaying="y",
                                        side="right"
                                    ),
                                    xaxis_title="Date",
                                    height=600,
                                    showlegend=True,
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display a table with the data
                                st.subheader("Price Data")
                                
                                # Create DataFrame for display
                                display_data = [{
                                    'Date': price_data['date'],
                                    'Open': f"${price_data['open']:.2f}",
                                    'High': f"${price_data['high']:.2f}",
                                    'Low': f"${price_data['low']:.2f}",
                                    'Close': f"${price_data['close']:.2f}",
                                    'Volume': f"{price_data['volume']:,}",
                                    'Daily Change': f"{((price_data['close'] - price_data['open']) / price_data['open'] * 100):+.2f}%"
                                }]
                                
                                # Add following days data
                                for day in price_data.get('following_days', []):
                                    daily_change = ((day['close'] - day['open']) / day['open'] * 100)
                                    display_data.append({
                                        'Date': day['date'],
                                        'Open': f"${day['open']:.2f}",
                                        'High': f"${day['high']:.2f}",
                                        'Low': f"${day['low']:.2f}",
                                        'Close': f"${day['close']:.2f}",
                                        'Volume': f"{day['volume']:,}",
                                        'Daily Change': f"{daily_change:+.2f}%"
                                    })
                                
                                st.dataframe(pd.DataFrame(display_data), use_container_width=True)
                                
                            except Exception as e:
                                logger.error(f"Error creating price chart: {str(e)}")
                                st.error(f"Error displaying price data: {str(e)}")
                        else:
                            st.warning("Incomplete price data available for chart display.")
        elif analyze_button or fetch_market_button or fetch_news_button or fetch_detailed_button:
             # This case handles when a button was clicked but no data was retrieved (errors handled above)
             pass # No need to display anything if no data is available and errors are shown
    

if __name__ == "__main__":
    main() 