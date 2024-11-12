# Data Collection Scripts README

## Overview

This document outlines the plan for collecting recent factual data from general, domain-neutral sources to fine-tune our language model. The data collected should adhere to the following criteria:

- Accurate and up-to-date
- Neutral and unbiased
- Legally and ethically permissible to collect and use

## Data Sources

The following websites and resources have been identified as suitable for data collection based on our compliance check:

1. **Public Library of Science (PLOS)**
   - **Website:** [https://www.plos.org/](https://www.plos.org/)
   - **Rationale:** Publishes open-access scientific research under Creative Commons licenses, allowing non-commercial use with proper attribution. Provides recent scientific discoveries and studies across various disciplines.

2. **World Health Organization (WHO)**
   - **Website:** [https://www.who.int/](https://www.who.int/)
   - **Rationale:** Content is available for non-commercial, educational purposes with attribution. Offers up-to-date global health information, reports, and statistics. Recognized as a neutral and authoritative source.

3. **United Nations (UN)**
   - **Website:** [https://www.un.org/](https://www.un.org/)
   - **Rationale:** Provides content for non-commercial use with proper attribution. Covers international news, official statements, and global initiatives. Neutral and comprehensive coverage of worldwide events.

4. **NewsAPI.org**
   - **Website:** [https://newsapi.org/](https://newsapi.org/)
   - **Rationale:** Aggregates news articles from various sources through an API. Simplifies data retrieval with structured JSON responses. Usage is permitted under specific terms; ensure compliance with their Terms of Use.

5. **GDELT Project**
   - **Website:** [https://www.gdeltproject.org/](https://www.gdeltproject.org/)
   - **Rationale:** Offers open-access data on global events, people, and themes. Provides data in machine-readable formats suitable for analysis. Ideal for capturing recent events and trends.

6. **Open Data Portals**
   - **a. Data.gov**
     - **Website:** [https://www.data.gov/](https://www.data.gov/)
     - **Rationale:** U.S. government’s open data portal. Contains datasets on a wide range of topics like agriculture, climate, and education. Data is public domain, free to use without restrictions.
   - **b. European Union Open Data Portal**
     - **Website:** [https://data.europa.eu/euodp/en/home](https://data.europa.eu/euodp/en/home)
     - **Rationale:** Provides access to datasets from EU institutions. Covers economics, health, and environmental data. Free to use with attribution, promoting transparency and innovation.

7. **Common Crawl**
   - **Website:** [https://commoncrawl.org/](https://commoncrawl.org/)
   - **Rationale:** Contains extensive web crawl data that can be processed to extract relevant information. Data is publicly available for research and analysis. Enables extraction of factual content from a broad range of websites.

## Data Collection Strategy

### APIs and Open Datasets

- Utilize APIs like NewsAPI.org and datasets from open data portals.
- Ensure compliance with each API's usage policies and terms.

### Web Scraping

- Scrape content only from websites that permit it, such as PLOS, WHO, and UN.
- Adhere to each site's robots.txt file and terms of service.
- Implement polite scraping practices, including rate limiting and respectful user-agent identification.

### Data Verification

- Cross-reference information across multiple sources.
- Prioritize data that is reported consistently by reputable organizations.

## Ensuring Data Neutrality and Objectivity

To ensure the collected data is factual and free from bias:

### Subjectivity Analysis

- Use natural language processing techniques to assess the subjectivity of the text.
- Retain texts classified as objective, filtering out subjective content.

### Sentiment Analysis

- Apply sentiment analysis models to detect emotionally charged language.
- Exclude content with strong positive or negative sentiments.

### Content Filtering

- Remove texts containing politically biased or opinionated statements.
- Focus on neutral reporting and factual descriptions.

## Data Formatting and Storage

### Data Fields

- **source:** Origin of the content (e.g., website or dataset name).
- **date:** Publication date of the text.
- **text:** The factual content or statement.
- **url:** URL of the original content (if applicable).
- **subjectivity_score:** Numerical score indicating the level of subjectivity.
- **sentiment_score:** Numerical score indicating the sentiment polarity.

### Data Format

- Store data in CSV or JSON files for ease of processing.
- Ensure consistent encoding (e.g., UTF-8) to handle special characters.

### Data Collection

1. Sign up for a NewsAPI key at https://newsapi.org/
2. Create a `.env` file in the project root
3. Add your NewsAPI key to the `.env` file:
   ```
   NEWS_API=your_api_key_here
   ```
4. Run the data collection script:
   ```bash
   python scripts/data/collect_data.py
   ```