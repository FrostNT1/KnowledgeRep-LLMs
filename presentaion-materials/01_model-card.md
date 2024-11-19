# Model Card

## Model Details

- **Model Name**: Llama 3.2 MLP-Fine-Tuned Model
- **Model Version**: 1.0
- **Model Type**: Causal Language Model
- **Architecture**: Llama 3.2 with 1B parameters, fine-tuned using 4-bit quantization (bnb-4bit)
- **Fine-Tuned From**: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **License**: MIT License
- **Author**: [Your Name or Organization]
- **Contact Information**: [Your Email or Contact Method]

## Model Description

This model is a fine-tuned version of the Llama 3.2 1B parameter language model, specifically fine-tuned on new data to investigate the hypothesis that **MLP layers in LLMs store world knowledge**. The model was fine-tuned by updating only the MLP layers while keeping the attention layers frozen.

## Intended Use

- **Primary Use Case**: Research and educational purposes to explore knowledge representation in large language models.
- **Potential Applications**:
  - Investigating efficient methods for updating LLMs with new information.
  - Studying the role of different neural network components in knowledge storage.
- **Restrictions on Use**:
  - Not intended for commercial applications.
  - Users should be aware of limitations due to the small fine-tuning dataset.

## Factors

- **Languages**: English
- **Domains**: Data science, as sourced from Vanderbilt DSI news and blogs.
- **Tasks**: Question answering, language generation.

## Training Details

- **Training Data**: The model was fine-tuned on the **Vanderbilt DSI QA Dataset**, consisting of 581 question-answer pairs generated from scraped articles.
- **Data Source**: Publicly available content from the Vanderbilt Data Science Institute website.
- **Fine-Tuning Procedure**:
  - **Layer Freezing**: Only the MLP layers were fine-tuned; attention layers were frozen.
  - **Hardware Used**: Google Colab with NVIDIA T4 GPU / A100.
  - **Optimizer**: AdamW (assumed if not specified).
  - **Loss Function**: Cross-Entropy Loss.
  - **Batch Size**: Not specified.
  - **Learning Rate**: Not specified.
  - **Number of Epochs**: Not specified.
- **Quantization**: 4-bit quantization using `bitsandbytes` (bnb-4bit) to reduce memory footprint.

## Evaluation

- **Evaluation Dataset**: A portion of the Vanderbilt DSI QA Dataset held out for testing.
- **Metrics**:
  - **Exact Match (EM) Score**: Measures the percentage of predictions that match the ground truth answers exactly.
  - **F1 Score**: Harmonic mean of precision and recall at the token level.
  - **Human Evaluation**: Assessments of correctness and relevance by human evaluators.
- **Results**:
  - The model demonstrated improved performance in recalling new information compared to models fine-tuned on attention layers or not fine-tuned.
  - Specific scores are not provided.

## Limitations

- **Data Limitations**:
  - The model was fine-tuned on a small dataset (581 QA pairs), which may limit its generalization capabilities.
  - The data is domain-specific to Vanderbilt DSI content.
- **Performance Limitations**:
  - May not perform well on tasks outside the domain of the fine-tuning data.
  - Potential for overfitting due to limited training data.

## Ethical Considerations

- **Biases**:
  - The model may reflect any biases present in the training data.
- **Usage Cautions**:
  - Users should be cautious when applying the model to critical tasks.
  - Not suitable for applications requiring high accuracy or broad general knowledge.

## How to Use

To use the model, you can load it using the Hugging Face Transformers library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('path_to_your_model')
model = AutoModelForCausalLM.from_pretrained('path_to_your_model')

# Generate text
input_text = "Your prompt here"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
```

## License

This model is released under the MIT License.

## Citation

If you use this model in your research, please cite it as:

```
@misc{your_name_2023_llama_mlp_finetuned,
  title={Llama 3.2 MLP-Fine-Tuned Model},
  author={Your Name},
  year={2023},
  howpublished={\url{https://github.com/your_repo}},
}
```

## Acknowledgments

- Llama 3.2 Model: Thanks to the developers of the Llama 3.2 model.
- unsloth: For providing the base model unsloth/Llama-3.2-1B-Instruct-bnb-4bit.
- Vanderbilt Data Science Institute: For the publicly available data used in fine-tuning.
