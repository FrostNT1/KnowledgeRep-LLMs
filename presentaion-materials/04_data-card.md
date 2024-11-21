# Data Card: Vanderbilt DSI QA Dataset

## Dataset Overview

- **Name**: Vanderbilt DSI QA Dataset
- **Version**: 1.0
- **Type**: Question-Answer Pairs
- **Domain**: Academic/Educational
- **Language**: English

## Data Collection

### Source Data
- **Source**: Vanderbilt Data Science Institute website
- **Content Type**: News articles and web content
- **Collection Method**: Web scraping using BeautifulSoup4 (bs4)
- **Total Articles**: 171 unique articles

### Dataset Creation
- **Generation Method**: GPT-assisted question-answer pair generation
- **Generation Strategy**: Each article was processed by ChatGPT to create approximately 10 quiz-style questions and answers
- **Quality Assurance**: ChatGPT was instructed to read articles and generate questions as if preparing an educational quiz

## Dataset Composition

### Size and Splits
- **Total QA Pairs**: 1,710 pairs
- **Training Set**: 1,368 pairs (80%)
- **Test Set**: 342 pairs (20%)
- **Validation Set**: Not included

### Data Format
- **Storage Formats**:
  - Raw data: CSV/Parquet
  - Processed data: Hugging Face Dataset (.hf)
- **Fields**:
  - topic: Source article topic/content
  - question: Generated question
  - answer: Corresponding answer
- **Final Format**: Hugging Face Dataset class

## Data Processing

### Processing Pipeline
1. Web scraping of articles using bs4
2. Basic text cleaning
3. ChatGPT-assisted QA pair generation
4. Conversion to Parquet format
5. Final conversion to Hugging Face Dataset format

### Data Quality
- **Quality Control**: 
  - Semantic coherence ensured through ChatGPT's quiz-style generation
  - Logical consistency between questions and source content
  - Direct relationship between questions and article content

## Intended Uses

- Fine-tuning language models
- Question-answering systems development
- Educational content assessment
- Research purposes

## Limitations

- Limited to Vanderbilt DSI domain content
- Questions generated artificially (not from human users)
- Dataset size relatively small (1,710 pairs)

## Ethical Considerations

### Privacy
- Data sourced from publicly available content
- No personal or sensitive information included

### Usage Rights
- Academic and research use
- Based on publicly available institutional content

## Distribution

- **Format**: Hugging Face Dataset
- **Access**: Through project repository
- **Updates**: Version 1.0 (static dataset)

## Citation

```bibtex
@misc{tyagi_2023_vanderbilt_dsi_qa,
  title={Vanderbilt DSI QA Dataset},
  author={Tyagi, Shivam},
  year={2023},
  publisher={Vanderbilt University}
}
```

## Maintenance

- **Creator**: Shivam Tyagi
- **Contact**: shivamtyagi@vanderbilt.edu
- **Last Updated**: 2023 November 20
