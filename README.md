# Understanding World Knowledge Representation in LLMs

This research project investigates the role of MLP layers in storing and representing world knowledge within Large Language Models (LLMs).

## Project Overview

Large Language Models (LLMs) have revolutionized natural language processing by leveraging architectures that combine attention mechanisms and Multilayer Perceptron (MLP) layers. This project aims to investigate whether the MLP layers in LLMs are the primary repositories of world knowledge—encompassing factual information about entities, events, and relationships.

### Research Questions

1. Are MLP layers the primary storage mechanism for world knowledge in LLMs?
2. How do attention mechanisms interact with stored knowledge?
3. Can we develop more efficient methods for updating LLMs with new information?

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers library
- Additional requirements in `requirements.txt`

### Installation

```bash
git clone https://github.com/FrostNT1/KnowledgeRep-LLMs.git
cd KnowledgeRep-LLMs
pip install -r requirements.txt
```

### Project Structure

- `src/`: Source code for the project
  - `data/`: Data loading and processing utilities
  - `models/`: Model-related code and utilities
  - `experiments/`: Experimental scripts and analysis
- `scripts/`: Automation and utility scripts
  - `data/`: Data preparation scripts
  - `training/`: Model training scripts
  - `analysis/`: Analysis and visualization scripts
  - `utils/`: Utility scripts
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration and visualization

```
KnowledgeRep-LLMs/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_utils.py
│   └── experiments/
│       ├── __init__.py
│       └── layer_analysis.py
├── scripts/
│   ├── data/
│   │   ├── download_datasets.sh
│   │   └── preprocess_data.py
│   ├── training/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── analysis/
│   │   ├── analyze_mlp_layers.py
│   │   └── visualize_results.py
│   └── utils/
│       ├── setup_environment.sh
│       └── cleanup.sh
├── tests/
│   └── __init__.py
└── notebooks/
    └── exploration.ipynb
```

## Usage

[To add specific usage instructions once implemented]

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-knowledge-2024,
  author = {Shivam Tyagi},
  title = {Understanding World Knowledge Representation in LLMs: Do MLP Layers Hold the Key?},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/FrostNT1/KnowledgeRep-LLMs}
}
```

## Contact

Shivam Tyagi - [st.shivamtyagi.01@gmail.com]

Project Link: [https://github.com/FrostNT1/KnowledgeRep-LLMs](https://github.com/FrostNT1/KnowledgeRep-LLMs)