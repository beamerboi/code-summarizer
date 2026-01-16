# Code Summarizer

A sequence-to-sequence Transformer model for generating natural language summaries of Python code snippets.

## Overview

This project implements a code summarization system that takes Python code as input and generates concise natural language descriptions of what the code does. The model is trained on the CodeSearchNet dataset and uses a custom Transformer encoder-decoder architecture built from scratch in PyTorch.

## Project Structure

```
ML/
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── preprocessing.py    # Code/text cleaning utilities
│   │   └── tokenizer.py        # BPE tokenizer wrapper
│   ├── models/
│   │   ├── transformer.py      # Main encoder-decoder model
│   │   ├── encoder.py          # Transformer encoder
│   │   ├── decoder.py          # Transformer decoder
│   │   └── attention.py        # Multi-head attention & components
│   ├── training/
│   │   ├── trainer.py          # Training loop with checkpointing
│   │   └── scheduler.py        # Warmup cosine learning rate scheduler
│   └── evaluation/
│       └── metrics.py          # BLEU, ROUGE, perplexity metrics
├── scripts/
│   ├── download_data.py        # Downloads CodeSearchNet dataset
│   └── prepare_data.py         # Preprocesses and tokenizes data
├── train.py                    # Main training script
├── evaluate.py                 # Model evaluation script
├── summarize.py                # Inference script
├── config.py                   # Hyperparameters and configuration
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- tokenizers>=0.15.0
- datasets>=2.14.0
- nltk>=3.8.0
- rouge-score>=0.1.2
- numpy>=1.24.0
- tqdm>=4.65.0

## Usage

### Step 1: Download Dataset

Download the CodeSearchNet Python dataset:

```bash
python scripts/download_data.py
```

This downloads ~450K code-docstring pairs from the CodeSearchNet dataset.

### Step 2: Prepare Data

Preprocess the data and train the tokenizer:

```bash
python scripts/prepare_data.py
```

This script:
- Cleans and normalizes code snippets
- Filters low-quality pairs
- Trains a BPE tokenizer on the training data
- Tokenizes and saves train/val/test splits

### Step 3: Train Model

Train the Transformer model:

```bash
python train.py
```

Training configuration (see `config.py`):
- Epochs: 15
- Batch size: 32
- Learning rate: 1e-4 with warmup + cosine decay
- Model: 4 encoder layers, 4 decoder layers, 256 hidden dim, 8 attention heads

The best model checkpoint is saved to `checkpoints/best_model.pt`.

### Step 4: Evaluate Model

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

Options:
- `--checkpoint PATH`: Path to model checkpoint (default: best_model.pt)
- `--split {val,test}`: Dataset split to evaluate on
- `--max-samples N`: Limit evaluation to N samples
- `--show-examples N`: Display N example outputs

### Step 5: Generate Summaries

Generate a summary for a code snippet:

```bash
python summarize.py --input "def add(x, y): return x + y"
```

Or summarize a Python file:

```bash
python summarize.py --file path/to/script.py
```

Options:
- `--input CODE`: Code snippet to summarize
- `--file PATH`: Path to Python file
- `--beam-size N`: Beam search width (default: 5)
- `--temperature T`: Sampling temperature (default: 1.0)

## Model Architecture

The model uses a standard Transformer encoder-decoder architecture:

**Encoder:**
- 4 Transformer layers
- 8 attention heads
- 256-dimensional embeddings
- 1024-dimensional feed-forward layers
- Sinusoidal positional encoding

**Decoder:**
- 4 Transformer layers
- 8 attention heads
- Multi-head self-attention with causal masking
- Multi-head cross-attention to encoder outputs
- Shared embedding and output projection

**Total Parameters:** ~32M

## Evaluation Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| **Cross-Entropy Loss** | Training objective |
| **Perplexity** | Exponential of average loss |
| **BLEU-1/2/3/4** | N-gram precision with brevity penalty |
| **ROUGE-1/2/L** | Recall-oriented overlap metrics |

## Configuration

Key hyperparameters in `config.py`:

```python
VOCAB_SIZE = 32000          # BPE vocabulary size
MAX_CODE_LENGTH = 256       # Maximum input code length
MAX_SUMMARY_LENGTH = 64     # Maximum output summary length

D_MODEL = 256               # Model dimension
N_HEADS = 8                 # Attention heads
N_ENCODER_LAYERS = 4        # Encoder depth
N_DECODER_LAYERS = 4        # Decoder depth
D_FF = 1024                 # Feed-forward dimension
DROPOUT = 0.1               # Dropout rate

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WARMUP_STEPS = 4000
EPOCHS = 15
LABEL_SMOOTHING = 0.1
```

## Dataset

This project uses the **CodeSearchNet** dataset (Python subset):
- ~450K Python functions with docstrings
- Source: GitHub repositories
- License: See CodeSearchNet documentation

The data is automatically downloaded via the HuggingFace `datasets` library.

## Expected Results Range

Typical performance range on the test set:
- BLEU-4: 15-20
- ROUGE-L: 30-40
- Perplexity: 10-20


## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Husain et al., "CodeSearchNet Challenge" (2019)
- PyTorch Documentation: https://pytorch.org/docs


