# Earnings Call Sentiment Analyzer

A sophisticated Python application that analyzes earnings call transcripts using advanced AI and natural language processing to provide insights into market sentiment and potential stock movements. This tool combines the power of Google's Gemini AI and Finnhub's financial data to deliver comprehensive analysis of earnings calls.

## Features

- **AI-Powered Analysis**: Utilizes Google's Gemini AI for intelligent transcript analysis
- **Sentiment Analysis**: Evaluates the emotional tone and sentiment of earnings calls
- **Market Impact Prediction**: Predicts potential stock movements based on earnings call content
- **News Integration**: Fetches relevant news articles using Finnhub API
- **Interactive Dashboard**: User-friendly Streamlit interface for easy interaction
- **Rate Limiting**: Built-in protection against API rate limits
- **Error Handling**: Robust error handling and logging system

## Tech Stack

- **Python 3.8+**
- **Core Libraries**:
  - `streamlit` – Interactive web interface
  - `google-generativeai` – Gemini AI integration
  - `finnhub-python` – Financial data and news
  - `python-dotenv` – Environment variable management
  - `pandas` – Data manipulation
  - `yfinance` – Stock price data
  - `nltk` – Natural language processing
  - `matplotlib` & `seaborn` – Data visualization

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Gemini API key
- Finnhub API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/earnings-call-sentiment-analyzer.git
cd earnings-call-sentiment-analyzer
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys and configuration:
```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# API Rate Limits
FINNHUB_RATE_LIMIT=60  # calls per minute
GEMINI_RATE_LIMIT=60   # calls per minute

# Environment
ENVIRONMENT=development  # development, production
```

## How It Works

1. **Transcript Analysis**:
   - Input: Earnings call transcript text
   - Processing: Gemini AI analyzes the content
   - Output: Concise summary and sentiment score

2. **Sentiment Analysis**:
   - Evaluates the emotional tone of the transcript
   - Generates a sentiment score between -1 (negative) and 1 (positive)
   - Provides detailed sentiment breakdown

3. **Market Impact**:
   - Analyzes potential stock movement (UP/DOWN/FLAT)
   - Fetches historical stock data for context
   - Integrates with news sentiment

4. **News Integration**:
   - Fetches relevant news articles
   - Analyzes news sentiment
   - Provides context for market impact

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the Interface:
   - Enter or paste the earnings call transcript
   - Click "Analyze" to process the transcript
   - View the analysis results:
     - Transcript summary
     - Sentiment score
     - Predicted market impact
     - Related news articles

## Project Structure

```
earnings-call-sentiment-analyzer/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── README.md                # Documentation
│
├── api_integration/         # API client implementations
│   ├── __init__.py
│   ├── gemini_client.py     # Gemini AI integration
│   ├── finnhub_client.py    # Finnhub data integration
│   └── finnhub_search_client.py  # Company search
│
└── transcripts/             # Sample transcripts
```

## Security

- API keys are stored in environment variables
- Rate limiting implemented for API protection
- Error handling for API failures
- Secure configuration management

## 🛠️ Development

### Adding New Features

1. Create a new branch for your feature
2. Implement the feature
3. Add appropriate tests
4. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Include error handling

## Notes

- The application requires valid API keys to function
- Rate limits are enforced to prevent API abuse
- All analysis is performed in real-time
- Results may vary based on API availability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for advanced language processing
- Finnhub for financial data and news
- Streamlit for the web interface framework 