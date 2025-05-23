import requests
from bs4 import BeautifulSoup
from typing import Optional
import os

def download_transcript(url: str) -> str:
    """
    Download and extract text from a transcript URL.
    
    Args:
        url (str): URL of the transcript
        
    Returns:
        str: Extracted transcript text
        
    Raises:
        Exception: If download or parsing fails
    """
    try:
        # Download the page
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download transcript: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to parse transcript: {str(e)}")

def read_local_file(filepath: str) -> str:
    """
    Read a local transcript file.
    
    Args:
        filepath (str): Path to the transcript file
        
    Returns:
        str: File contents
        
    Raises:
        Exception: If file reading fails
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
            
    except Exception as e:
        raise Exception(f"Failed to read file: {str(e)}") 