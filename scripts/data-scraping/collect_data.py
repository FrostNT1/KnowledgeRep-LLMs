import os
import logging
from datetime import datetime
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
from dateutil import parser
from dotenv import load_dotenv
import json
from urllib.parse import quote
import xmltodict
import sodapy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API key from .env file
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API')
if not NEWS_API_KEY:
    raise ValueError("NEWS_API key not found in .env file")
DATAGOV_API = os.getenv('DATAGOV_API') 
if not DATAGOV_API:
    raise ValueError("DATAGOV_API key not found in .env file")

# Thresholds for subjectivity and sentiment
SUBJECTIVITY_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.3
START_DATE = (datetime.now() - pd.DateOffset(months=1)).strftime('%Y-%m-%d')

# Add these new constants
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
EU_DATA_BASE_URL = "https://data.europa.eu/api/hub/search/datasets"

def parse_date(date_string: str) -> str:
    """Parse and format date strings."""
    return parser.parse(date_string).strftime('%Y-%m-%d')

def analyze_text(text: str) -> tuple[float, float]:
    """Analyze subjectivity and sentiment of the text."""
    blob = TextBlob(text)
    return blob.sentiment.subjectivity, blob.sentiment.polarity

def fetch_newsapi_articles() -> List[Dict[str, Any]]:
    """Fetch articles from NewsAPI."""
    url = f'https://newsapi.org/v2/everything'
    params = {
        'q': 'latest',
        'language': 'en',
        'from': START_DATE,
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    
    logger.debug(f"NewsAPI Response: {response.json()}")
    articles = response.json().get('articles', [])
    logger.info(f"NewsAPI returned {len(articles)} articles")
    
    data = []
    threshold_filtered = 0
    
    for article in articles:
        date = article.get('publishedAt', '')
        source = article['source']['name']
        text = article.get('content', '')
        url = article.get('url', '')
        subjectivity_score, sentiment_score = analyze_text(text)
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            threshold_filtered += 1
            data.append({
                'source': source,
                'date': parse_date(date),
                'text': text,
                'url': url,
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"NewsAPI: {len(articles)} total, {threshold_filtered} passed threshold filter")
    return data

def fetch_plos_articles() -> List[Dict[str, Any]]:
    """Scrape recent articles from PLOS."""
    url = 'https://journals.plos.org/plosone/browse/new'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = soup.find_all('div', class_='article')
    logger.info(f"PLOS returned {len(articles)} articles")
    
    data = []
    threshold_filtered = 0
    
    for article in articles:
        date_text = article.find('span', class_='pubdate').get_text(strip=True)
        date = parse_date(date_text)
        title = article.find('h2').get_text(strip=True)
        text = title
        link = article.find('a', href=True)['href']
        subjectivity_score, sentiment_score = analyze_text(text)
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            threshold_filtered += 1
            data.append({
                'source': 'PLOS',
                'date': date,
                'text': text,
                'url': link,
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"PLOS: {len(articles)} total, {threshold_filtered} passed threshold filter")
    return data

def fetch_who_articles() -> List[Dict[str, Any]]:
    """Scrape recent news releases from WHO."""
    url = 'https://www.who.int/news-room/releases'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = soup.find_all('div', class_='list-view--item vertical-list-item')
    logger.info(f"WHO returned {len(articles)} articles")
    
    data = []
    threshold_filtered = 0
    
    for article in articles:
        title = article.find('a').get_text(strip=True)
        date_text = article.find('span', class_='timestamp').get_text(strip=True)
        date = parse_date(date_text)
        link = "https://www.who.int" + article.find('a')['href']
        subjectivity_score, sentiment_score = analyze_text(title)
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            threshold_filtered += 1
            data.append({
                'source': 'WHO',
                'date': date,
                'text': title,
                'url': link,
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"WHO: {len(articles)} total, {threshold_filtered} passed threshold filter")
    return data

def fetch_un_articles() -> List[Dict[str, Any]]:
    """Scrape recent news from the United Nations."""
    url = 'https://www.un.org/press/en'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = soup.find_all('div', class_='views-row')
    logger.info(f"UN returned {len(articles)} articles")
    
    data = []
    threshold_filtered = 0
    
    for article in articles:
        title = article.find('a').get_text(strip=True)
        date_text = article.find('span', class_='date-display-single').get_text(strip=True)
        date = parse_date(date_text)
        link = "https://www.un.org" + article.find('a')['href']
        subjectivity_score, sentiment_score = analyze_text(title)
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            threshold_filtered += 1
            data.append({
                'source': 'UN',
                'date': date,
                'text': title,
                'url': link,
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"UN: {len(articles)} total, {threshold_filtered} passed threshold filter")
    return data

def fetch_gdelt_articles() -> List[Dict[str, Any]]:
    """Fetch recent articles from GDELT Project."""
    params = {
        'format': 'json',
        'maxrecords': 250,
        'timespan': '1440',  # Last 24 hours
        'mode': 'artlist',  # Add this parameter
        'format': 'json'    # Note: 'format' is duplicated in original
    }
    
    try:
        response = requests.get(GDELT_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Debug logging
        logger.debug(f"GDELT Response Status: {response.status_code}")
        logger.debug(f"GDELT Response Content: {response.text[:500]}...")  # First 500 chars
        
        articles = response.json().get('articles', [])
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from GDELT: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing GDELT response: {str(e)}")
        logger.error(f"Response content: {response.text[:500]}...")
        return []
    
    data = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('seentext', '')}"
        date = parse_date(article.get('seendate', ''))
        subjectivity_score, sentiment_score = analyze_text(text)
        
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            data.append({
                'source': 'GDELT',
                'date': date,
                'text': text,
                'url': article.get('url', ''),
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"GDELT: {len(articles)} total, {len(data)} passed threshold filter")
    return data

def fetch_datagov_articles() -> List[Dict[str, Any]]:
    """Fetch recent datasets from Data.gov."""
    client = sodapy.Socrata("data.gov", DATAGOV_API)
    
    # Fetch recent datasets
    results = client.get("7g6j-rrh5", limit=100, order="modification_date DESC")
    
    data = []
    for result in results:
        text = f"{result.get('title', '')} {result.get('description', '')}"
        date = parse_date(result.get('modification_date', ''))
        subjectivity_score, sentiment_score = analyze_text(text)
        
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            data.append({
                'source': 'Data.gov',
                'date': date,
                'text': text,
                'url': result.get('landingPage', ''),
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"Data.gov: {len(results)} total, {len(data)} passed threshold filter")
    return data

def fetch_eu_data_articles() -> List[Dict[str, Any]]:
    """Fetch recent datasets from EU Open Data Portal."""
    params = {
        'limit': 100,
        'sort': 'modified',
        'order': 'desc',
        'format': 'json'
    }
    
    response = requests.get(EU_DATA_BASE_URL, params=params)
    results = response.json().get('result', {}).get('results', [])
    
    data = []
    for result in results:
        text = f"{result.get('title', '')} {result.get('description', '')}"
        date = parse_date(result.get('modified', ''))
        subjectivity_score, sentiment_score = analyze_text(text)
        
        if subjectivity_score <= SUBJECTIVITY_THRESHOLD and abs(sentiment_score) <= SENTIMENT_THRESHOLD:
            data.append({
                'source': 'EU Open Data',
                'date': date,
                'text': text,
                'url': result.get('landingPage', ''),
                'subjectivity_score': subjectivity_score,
                'sentiment_score': sentiment_score
            })
    
    logger.info(f"EU Data Portal: {len(results)} total, {len(data)} passed threshold filter")
    return data

def main():
    try:
        logger.info("Starting data collection process...")

        # Collect data from all sources
        logger.info("Fetching data from NewsAPI...")
        newsapi_data = fetch_newsapi_articles()

        logger.info("Fetching data from PLOS...")
        plos_data = fetch_plos_articles()

        logger.info("Fetching data from WHO...")
        who_data = fetch_who_articles()

        logger.info("Fetching data from UN...")
        un_data = fetch_un_articles()

        # New data collection
        logger.info("Fetching data from GDELT...")
        gdelt_data = fetch_gdelt_articles()
        
        logger.info("Fetching data from Data.gov...")
        datagov_data = fetch_datagov_articles()
        
        logger.info("Fetching data from EU Open Data Portal...")
        eu_data = fetch_eu_data_articles()

        # Combine all data into one list
        all_data = newsapi_data + plos_data + who_data + un_data + \
                   gdelt_data + datagov_data + eu_data

        # Convert data to DataFrame
        df = pd.DataFrame(all_data)

        if df.empty:
            logger.warning("No articles collected. DataFrame is empty. Skipping save operation.")
        else:
            # Use DATA_PATH from .env
            output_dir = os.getenv("DATA_PATH")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a timestamped filename for Parquet format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f'collected_data_{timestamp}.parquet')
            
            # Save data to Parquet
            df.to_parquet(output_path, index=False)
            
            logger.info(f"Data collection complete. Saved to '{output_path}'")
            logger.info(f"Total articles collected: {len(df)}")

    except Exception as e:
        logger.error(f"An error occurred during data collection: {str(e)}")
        raise

if __name__ == '__main__':
    main()
