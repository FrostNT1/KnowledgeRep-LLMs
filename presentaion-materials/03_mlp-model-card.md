# Model Card: Llama 3.2 MLP-Only Fine-Tuned Model

## Model Details

- **Model Name**: Llama 3.2 MLP-Only Fine-Tuned Model
- **Model Version**: 1.0
- **Model Type**: Causal Language Model
- **Architecture**: Llama 3.2 with 1B parameters, fine-tuned using 4-bit quantization (bnb-4bit)
- **Fine-Tuned From**: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **License**: MIT License
- **Author**: Shivam Tyagi
- **Contact Information**: shivamtyagi@vanderbilt.edu

## Model Description

This model is a fine-tuned version of the Llama 3.2 1B parameter language model, specifically fine-tuned on Vanderbilt DSI data. The model was fine-tuned using LoRA to update only the MLP layers, leaving attention layers frozen.

## Intended Use

- **Primary Use Case**: Research and educational purposes
- **Potential Applications**:
  - Question answering about Vanderbilt DSI content
  - Text generation related to DSI topics
- **Restrictions on Use**:
  - Not intended for commercial applications
  - Users should be aware of limitations due to the domain-specific training

## Training Details

### Model Architecture
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: 
  - MLP only: gate_proj, up_proj, down_proj
  - Attention layers remain frozen
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0
  - Bias: none

### Training Hyperparameters
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (effective batch size: 8 with gradient accumulation)
- **Gradient Accumulation Steps**: 4
- **Maximum Steps**: 60
- **Warmup Steps**: 5
- **Weight Decay**: 0.01
- **Optimizer**: AdamW 8-bit
- **Learning Rate Schedule**: Linear
- **Maximum Sequence Length**: 2048

### Training Infrastructure
- **Hardware**: Google Colab with NVIDIA T4 GPU
- **Quantization**: 4-bit quantization using `bitsandbytes`
- **Memory Optimizations**: 
  - Gradient checkpointing enabled
  - Expandable CUDA segments

## Limitations

- **Domain Limitations**: 
  - Trained specifically on Vanderbilt DSI content
  - May not perform well on general domain questions
- **Technical Limitations**:
  - 4-bit quantization may impact model precision
  - Limited by the base model's 1B parameter size
  - Only MLP layers were fine-tuned, potentially affecting attention-based capabilities

## License

This model is released under the MIT License.

## Citation

```
@misc{tyagi_2023_llama_mlp_finetuned,
  title={Llama 3.2 MLP-Only Fine-Tuned Model},
  author={Tyagi, Shivam},
  year={2023},
  publisher={Vanderbilt University}
}
```

## Acknowledgments

- Llama 3.2 Model: Meta AI
- unsloth: For providing the base 4-bit quantized model
- Vanderbilt Data Science Institute: For the training data
