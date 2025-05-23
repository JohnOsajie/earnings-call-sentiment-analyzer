# Earnings Call Sentiment Analyzer

A Python application that analyzes the sentiment of earnings call transcripts using natural language processing (NLP). This tool helps investors and analysts understand the emotional tone and sentiment trends in corporate earnings calls.

## 🧰 Tech Stack

- **Python 3.8+**
- **Core Libraries**:
  - `requests` – for downloading transcripts
  - `beautifulsoup4` – for HTML parsing
  - `nltk` – for sentiment analysis with VADER
  - `matplotlib` – for sentiment visualization
  - `pandas` – for data organization
- **Optional**: `streamlit` for dashboard interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/earnings-call-sentiment-analyzer.git
cd earnings-call-sentiment-analyzer
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -m nltk.downloader vader_lexicon
```

## 📄 How It Works

1. **Input**: User provides either:
   - URL to an earnings call transcript
   - Local .txt file containing the transcript

2. **Processing**:
   - Downloads and parses the transcript (if URL provided)
   - Splits text into paragraphs/sentences
   - Analyzes sentiment using VADER

3. **Analysis**:
   - Calculates sentiment scores:
     - Positive
     - Neutral
     - Negative
     - Compound (overall sentiment)

4. **Output**:
   - Generates sentiment visualization
   - Exports results to CSV
   - Displays console summary

## 🧪 Usage

### Command Line Interface

```bash
# Analyze from a transcript URL
python main.py --url https://example.com/earnings-call.html

# Analyze from a local .txt file
python main.py --file transcripts/apple_q1_2024.txt
```

### Example Output

```yaml
Overall Compound Sentiment: 0.45 (Positive)
Positive: 63%, Neutral: 30%, Negative: 7%
```

The analysis generates:
- `sentiment_plot.png` – Visual graph of sentiment scores
- `sentiment_scores.csv` – Detailed sentiment breakdown

## 📁 Project Structure

```
earnings-call-sentiment-analyzer/
│
├── main.py              # Entry point
├── analyzer.py          # Sentiment analysis logic
├── utils.py            # Web/text processing helpers
├── requirements.txt    # Project dependencies
├── README.md          # Documentation
└── transcripts/       # Sample transcripts
    └── sample.txt
```

## 📝 Notes

- All file operations use UTF-8 encoding
- Input files should be plain text (.txt) format
- Avoids binary file operations to prevent null byte issues

## 📦 Dependencies

```
requests
beautifulsoup4
nltk
pandas
matplotlib
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 