# SeqBench: Sequence Model Benchmarking

This project benchmarks sequence models (RNN, Transformer, Mamba-like) with Hydra, Aim, and MLflow, including CUDA event-based latency sweeps.

## Features

- **Multiple Model Architectures**: RNN, Transformer, Mamba-like models
- **Hydra Configuration**: Flexible configuration management
- **Experiment Tracking**: Integration with Aim and MLflow
- **CUDA Event-Based Latency**: Precise timing measurements using CUDA events
- **Synthetic Data Generation**: Built-in synthetic data generation for quick testing
- **Hugging Face Datasets**: Support for real datasets like TinyStories from Hugging Face

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Configure Hugging Face Cache (Optional)

To set a custom cache directory for Hugging Face datasets and models:

```bash
# Or manually set environment variables
export HF_HOME="/data/hf_cache"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"
export TRANSFORMERS_CACHE="/data/hf_cache/transformers"
```

To make this permanent, add these lines to your `~/.bashrc` or `~/.zshrc`.

### Start Logging Services

```bash
# Start Aim dashboard (port 43800)
aim up --port 43800 &

# Start MLflow UI (port 5000)
mlflow ui --port 5000 &
```

### Run Training

```bash
# Train with synthetic data (default)
python run.py model=rnn mode=train

# Train with TinyStories dataset from Hugging Face
python run.py model=rnn mode=train data=tinystories
```

### Run Latency Benchmark

```bash
python run.py model=rnn mode=latency
```

## Results

Results appear in:
- **Outputs**: Hydra stores configs under `outputs/DATE_TIME/`
- **Aim Dashboard**: Access at `http://localhost:43800`
- **MLflow UI**: Access at `http://localhost:5000`

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `conf/config.yaml`: Main configuration
- `conf/model/*.yaml`: Model-specific configurations
- `conf/trainer/default.yaml`: Training hyperparameters
- `conf/bench/default.yaml`: Benchmarking parameters
- `conf/data/synthetic.yaml`: Synthetic data generation parameters
- `conf/data/tinystories.yaml`: TinyStories dataset configuration

### Example: Change Model

```bash
python run.py model=transformer mode=train
```

### Example: Override Parameters

```bash
python run.py model=rnn mode=train trainer.max_steps=500 trainer.lr=1e-3
```

## Project Structure

```
seqbench/
├── run.py                 # Main entry point
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── conf/                 # Hydra configuration files
│   ├── config.yaml
│   ├── model/
│   │   ├── rnn.yaml
│   │   ├── transformer.yaml
│   │   └── mamba.yaml
│   ├── trainer/
│   │   └── default.yaml
│   ├── bench/
│   │   └── default.yaml
│   └── data/
│       └── synthetic.yaml
└── src/
    ├── models/           # Model implementations
    │   ├── rnn_pytorch.py
    │   ├── transformer_custom.py
    │   ├── mamba_custom.py
    │   └── gru_custom.py
    └── utils/            # Utility functions
        ├── bench.py      # Latency benchmarking
        ├── logging.py    # Aim/MLflow loggers
        └── __init__.py
```

## Model Implementations

- **RNN**: Fully implemented using PyTorch's `nn.RNN`
- **Transformer**: Placeholder (TODO)
- **Mamba**: Placeholder (TODO)
- **GRU**: Placeholder (TODO)

## Acceptance Criteria

✅ `python run.py model=rnn mode=train` trains and prints loss  
✅ `python run.py model=rnn mode=latency` prints latency per token  
✅ Aim and MLflow both receive logs  
✅ Hydra stores configs under `outputs/DATE_TIME/`  
✅ Folder structure matches exactly as described

