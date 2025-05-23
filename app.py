import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import SentimentAnalyzer
from utils import download_transcript, read_local_file
import io
from datetime import datetime, timedelta
from api_integration import get_sentiment, process_sentiment_data

st.set_page_config(
    page_title="Earnings Call Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Earnings Call Sentiment Analyzer")
    st.write("Analyze the sentiment of earnings call transcripts using NLP")
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Sentiment Analysis", "API Sentiment"])
    
    with tab1:
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Transcript", "Paste Transcript", "Use Sample Transcript"]
        )
        
        content = None
        
        if input_method == "Upload Transcript":
            uploaded_file = st.file_uploader("Upload a transcript file", type=['txt'])
            if uploaded_file is not None:
                content = uploaded_file.getvalue().decode('utf-8')
                
        elif input_method == "Paste Transcript":
            st.write("Paste your earnings call transcript below:")
            pasted_content = st.text_area(
                "Transcript",
                height=300,
                help="Paste the earnings call transcript here. The text should include the dialogue between company executives and analysts."
            )
            if pasted_content:
                content = pasted_content
                
        else:  # Use Sample Transcript
            try:
                content = read_local_file("transcripts/sample.txt")
            except Exception as e:
                st.error(f"Error reading sample transcript: {str(e)}")
        
        if content:
            # Initialize analyzer
            analyzer = SentimentAnalyzer()
            
            # Analyze content
            with st.spinner("Analyzing sentiment..."):
                results = analyzer.analyze(content)
                
                # Display overall sentiment
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Compound Sentiment", f"{results['compound']:.2f}")
                with col2:
                    st.metric("Positive", f"{results['positive']*100:.0f}%")
                with col3:
                    st.metric("Neutral", f"{results['neutral']*100:.0f}%")
                with col4:
                    st.metric("Negative", f"{results['negative']*100:.0f}%")
                
                # Create visualization
                st.subheader("Sentiment Analysis Over Time")
                fig, ax = plt.subplots(figsize=(12, 6))
                df = pd.DataFrame(results['paragraphs'])
                ax.plot(df['paragraph'], df['compound'], label='Compound')
                ax.plot(df['paragraph'], df['positive'], label='Positive')
                ax.plot(df['paragraph'], df['negative'], label='Negative')
                ax.set_title('Sentiment Analysis Over Time')
                ax.set_xlabel('Paragraph Number')
                ax.set_ylabel('Sentiment Score')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                # Display detailed results
                st.subheader("Detailed Analysis")
                st.dataframe(
                    pd.DataFrame(results['paragraphs']).style.background_gradient(
                        subset=['compound', 'positive', 'negative', 'neutral'],
                        cmap='RdYlGn'
                    )
                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Save CSV
                    csv = pd.DataFrame(results['paragraphs']).to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="sentiment_scores.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Save plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot",
                        data=buf,
                        file_name="sentiment_plot.png",
                        mime="image/png"
                    )
    
    with tab2:
        st.subheader("API Sentiment Analysis")
        
        # Input for stock ticker and date
        col1, col2 = st.columns(2)
        with col1:
            api_ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
        with col2:
            api_date = st.date_input(
                "Select Date:",
                datetime.now() - timedelta(days=1)
            )
        
        if st.button("Fetch API Sentiment"):
            with st.spinner("Fetching sentiment data from API..."):
                try:
                    # Fetch sentiment data
                    sentiment_data = get_sentiment(api_ticker, api_date)
                    
                    # Process the data
                    processed_data = process_sentiment_data(sentiment_data)
                    
                    # Display sentiment metrics
                    st.subheader("Sentiment Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Sentiment Score",
                            f"{processed_data['sentiment_score']:.2f}",
                            delta=None
                        )
                    with col2:
                        st.metric(
                            "Buzz Score",
                            f"{processed_data['buzz_score']:.2f}",
                            delta=None
                        )
                    with col3:
                        st.metric(
                            "Category",
                            processed_data['sentiment_category'],
                            delta=None
                        )
                    
                    # Display summary
                    st.subheader("Summary")
                    st.write(processed_data['summary'])
                    
                    # Display article statistics if available
                    if 'article_stats' in processed_data:
                        st.subheader("Article Statistics")
                        stats = processed_data['article_stats']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Articles", stats['total_articles'])
                        with col2:
                            st.metric("Positive Articles", stats['positive_articles'])
                        with col3:
                            st.metric("Negative Articles", stats['negative_articles'])
                        with col4:
                            st.metric("Neutral Articles", stats['neutral_articles'])
                    
                    # Display articles if available
                    if processed_data['articles']:
                        st.subheader("Recent Articles")
                        for article in processed_data['articles']:
                            with st.expander(f"{article['title']} ({article['category']})"):
                                st.write(f"**Published:** {article['published']}")
                                st.write(f"**Summary:** {article['summary']}")
                                st.write(f"**URL:** {article['url']}")
                    
                except Exception as e:
                    st.error(f"Error fetching sentiment data: {str(e)}")

if __name__ == "__main__":
    main() 