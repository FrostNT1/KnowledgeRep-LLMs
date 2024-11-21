# Dataset Card

## Dataset Description

- **Dataset Name**: Vanderbilt DSI QA Dataset
- **Dataset Version**: 1.0
- **Dataset Link**: [Provide link if available]
- **Dataset Size**: 581 question-answer pairs
- **Languages**: English

### Summary

This dataset consists of question-answer pairs generated from articles and blog posts published on the Vanderbilt Data Science Institute (DSI) website. The data was collected to fine-tune language models for research on knowledge representation within LLMs.

## Dataset Composition

- **Data Instances**:

  Each data instance includes:

  - **Question**: A question generated from the content.
  - **Answer**: The corresponding answer extracted from the content.

- **Number of Instances**: 581

- **Data Fields**:

  - `question`: String containing the generated question.
  - `answer`: String containing the extracted answer.

- **Data Types**: Textual data.

## Data Collection Process

- **Source Data**:

  - **Origin**: Publicly available articles and blog posts from the Vanderbilt Data Science Institute website.
  - **Collection Method**: Web scraping with adherence to the website's terms of service.

- **Methodology**:

  - A script was used to scrape content from the Vanderbilt DSI website.
  - The scraped content was processed to generate question-answer pairs using pre-trained models:

    - **Question Generation**: Utilized the `valhalla/t5-small-qg-prepend` model.
    - **Answer Extraction**: Used the `distilbert-base-cased-distilled-squad` model.

  - **Script Details**:

    ```python
    from transformers import pipeline
    import torch
    from tqdm import tqdm  # Import tqdm for progress tracking

    # Load models
    question_generator = pipeline("text2text-generation", 
                                  model="valhalla/t5-small-qg-prepend",
                                  device=0 if torch.cuda.is_available() else -1)

    qa_extractor = pipeline("question-answering",
                            model="distilbert-base-cased-distilled-squad",
                            device=0 if torch.cuda.is_available() else -1)

    qa_pairs = []

    # Iterate over content with tqdm for progress tracking
    for text in tqdm(df['content'], desc="Processing Content"):
        try:
            # Split text into chunks of 300 characters with some overlap
            chunk_size = 300
            overlap = 50
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            
            unique_questions = set()
            
            for chunk in chunks[:5]:  # Limit processing to 5 chunks for efficiency
                input_text = f"generate questions: {chunk}"
                
                # Generate questions for the current chunk
                questions = question_generator(
                    input_text, 
                    max_length=512, 
                    truncation=True, 
                    num_beams=5, 
                    num_return_sequences=1  # Generate one question per chunk
                )
                
                # Collect unique questions
                for q in questions:
                    if '?' in q['generated_text']:
                        unique_questions.add(q['generated_text'])
            
            # Extract answers for up to 5 unique questions
            for question in list(unique_questions)[:5]:
                answer = qa_extractor(question=question, context=text)
                qa_pairs.append({"question": question, "answer": answer['answer']})
        except Exception as e:
            print(f"Error processing text: {e}")
            continue

    # Create a DataFrame from the QA pairs
    qa_df = pd.DataFrame(qa_pairs)
    ```

## Dataset Structure

- **Format**: Parquet file with two columns: `question` and `answer`.
- **Storage**: The dataset can be loaded using pandas:

  ```python
  import pandas as pd
  qa_df = pd.read_parquet('path_to_dataset.parquet')
  ```

## Data Instances

An example data instance:

- Question: "What event is being held at Vanderbilt DSI this month?"
- Answer: "Data Science Symposium"

## Intended Use
Intended Use:

- Fine-tuning language models for research on knowledge representation.
- Educational purposes in NLP and data science courses.

## Out-of-Scope Use

Commercial applications without proper permissions.
- Any use that violates the terms of service of the data sources.

## Ethical Considerations

- **Data Privacy**:

  - All data was collected from publicly available sources.
  - No personal or sensitive information is included.

- **Bias**:

  - The dataset is domain-specific to Vanderbilt DSI, which may introduce topical biases.

- **Licensing and Permissions**:

  - Users should ensure compliance with the data source's terms of service.
  - Redistribution of the dataset should respect any copyright restrictions.

- **License**:

  - The dataset is made available under the MIT License.

## Citation

If you use this dataset in your research, please cite it as:

```
@dataset{your_name_2023_vanderbilt_dsi_qa,
  title={Vanderbilt DSI QA Dataset},
  author={Your Name},
  year={2023},
  howpublished={\url{https://github.com/your_repo}},
}
```

## Acknowledgments

- Vanderbilt Data Science Institute: For providing the content used to create the dataset.

- **Model Developers**:

  - valhalla/t5-small-qg-prepend for question generation.
  - distilbert-base-cased-distilled-squad for answer extraction.