#!/usr/bin/env python3
import argparse
import sys
from analyzer import SentimentAnalyzer
from utils import download_transcript, read_local_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze sentiment of earnings call transcripts')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--url', help='URL of the earnings call transcript')
    group.add_argument('--file', help='Path to local transcript file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    try:
        # Get transcript content
        if args.url:
            content = download_transcript(args.url)
        else:
            content = read_local_file(args.file)
        
        # Initialize analyzer and process content
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze(content)
        
        # Display results
        analyzer.display_results(results)
        
        # Save results
        analyzer.save_results(results)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 