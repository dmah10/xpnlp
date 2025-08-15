<h2 align="center">
  <span>When Explainability Meets Privacy: An Investigation at the Intersection of Post-hoc Explainability and Differential Privacy in the Context of Natural Language Processing</span>
</h2>

This is the repository for the paper "When Explainability Meets Privacy: An Investigation at the Intersection of Post-hoc Explainability and Differential Privacy in the Context of Natural Language Processing" accepted to AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025

- Paper: (https://arxiv.org/abs/2508.10482)

## Overview

This project provides tools and methodologies for:
- Fine-tuning transformer models on various text classification datasets
- Generating and evaluating model explanations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda package manager

### Setup Steps

1. Clone the repository.

2. Create and activate the conda environment:
```bash
conda env create --file=environment.yml
conda activate [environment-name]
```

3. Install the custom Ferret package:
```bash
cd ferret
pip install .
```

4. Verify the Ferret installation:
```bash
python -c "from ferret import Benchmark; Benchmark.test_install()"
```

### Optional: Weights & Biases Setup
For experiment tracking (optional):
```bash
wandb login
```

## Project Structure

- `data/` - Dataset storage
- `src/` - Source code for core functionality
- `ferret/` - Custom explanation framework
- `evaluate/` - Evaluation metrics and utilities
- `results/` - Output storage for experiments

## Usage

### Quick Start

Run a test experiment to verify setup:
```bash
python main.py --batch-size 8 --epochs 1 --datasets trustpilot ag_news sst2 --models google-bert/bert-large-cased google-bert/bert-base-cased FacebookAI/roberta-large FacebookAI/roberta-base microsoft/deberta-large microsoft/deberta-base --project ignore --train --test-run --shrink
```

### Full Experiments

For complete experiments, remove the test flags:
```bash
python main.py --batch-size 8 --epochs 1 --datasets trustpilot ag_news sst2 --models google-bert/bert-large-cased google-bert/bert-base-cased FacebookAI/roberta-large FacebookAI/roberta-base microsoft/deberta-large microsoft/deberta-base --project <project_name> --train
```

### Important Parameters

- `--max-length <number>`: Filter inputs by character length
- `--max-tokens <number>`: Filter inputs by token count
- `--save`: Save generated explanations
- `--sensitivity`: Evaluate only sensitivity
- `--reps <number>`: Number of experiment repetitions
- `--project`: Weights & Biases project name (optional)

## Memory Management

To avoid out-of-memory errors on long inputs:
- Use `--max-length` to filter by character count
- Use `--max-tokens` to filter by token count
- Adjust `--batch-size` based on available GPU memory

## Model Support

The framework supports:
- BERT-based models
- RoBERTa models
- DeBERTa models

To add new models, simply include the Hugging Face model identifier in the `--models` argument.

## Full Command Line Arguments

### Core Parameters
- `--batch-size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 1)
- `--reps`: Number of experiment repetitions (default: 1)
- `--project`: Weights & Biases project name (default: "fairness")
- `--wandb`: Weights & Biases API key (optional)

### Data Management
- `--data-dir`: Directory to save the data (default: "./data")
- `--train-size`: Training set size (default: 20,000)
- `--test-size`: Test set size (default: 10,000)
- `--max-length`: Maximum input length in characters (default: 1,000,000)
- `--max-tokens`: Maximum input length in tokens (default: 512)
- `--min-length`: Minimum input length in characters (default: 0)
- `--split-idx`: Index of the dataset split (default: 2)

### Model and Dataset Selection
- `--models`: List of Hugging Face model names (default: ["huawei-noah/TinyBERT_General_4L_312D"])
- `--datasets`: List of datasets to use

### Experiment Control
- `--train`: Train the model 
- `--only-explain`: Generate and save explanations without evaluations
- `--save`: Save the trained model locally
- `--shrink`: Use reduced dataset size for testing
- `--test-run`: Run minimal experiment for testing
- `--random`: Assign random labels 
- `--sensitivity`: Compute sensitivity metrics
- `--soft-only`: Only compute soft evaluation metrics
- `--seeds`: List of random seeds for multiple runs (default: [1,2,3,4,5])


## Extensions

To add new **datasets**, write a function in `data.py` that loads the dataset as a Huggingface datasets `Dataset` object. There are no strict restrictions on what the columns should be labeled but the existing DP datasets follow the convention of having the inputs labeled `"text"` and the labels `"label"`.

To add new **models**, BERT-based models on Huggingface Hub should be directly usable just by setting the model string flag for `main.py`. Other models on the Hub might require slightly different setups depending on the specifics of the model. 
