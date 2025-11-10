"""Data loading utilities for synthetic and Hugging Face datasets."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Iterator, Tuple
import numpy as np


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick testing."""
    
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        """Initialize synthetic dataset.
        
        Args:
            vocab_size: Vocabulary size
            seq_len: Sequence length
            num_samples: Number of samples in dataset
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Target is shifted input (next token prediction)
        target_ids = torch.roll(input_ids, shifts=-1, dims=0)
        target_ids[-1] = input_ids[0]  # Wrap around last token
        return input_ids, target_ids


class HuggingFaceDataset(Dataset):
    """Wrapper for Hugging Face datasets."""
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int,
        text_column: str = 'text',
        max_samples: Optional[int] = None
    ):
        """Initialize Hugging Face dataset wrapper.
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            seq_len: Sequence length
            text_column: Column name containing text
            max_samples: Maximum number of samples to use
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.max_samples = max_samples or len(dataset)
        self.max_samples = min(self.max_samples, len(dataset))
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # Get text from dataset
        text = self.dataset[idx][self.text_column]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate or pad to seq_len
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Target is shifted input (next token prediction)
        target_ids = torch.roll(input_ids, shifts=-1, dims=0)
        target_ids[-1] = input_ids[0]  # Wrap around last token
        
        return input_ids, target_ids


def load_tinystories_dataset(
    seq_len: int,
    split: str = 'train',
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Dataset, int]:
    """Load TinyStories dataset from Hugging Face.
    
    Args:
        seq_len: Sequence length
        split: Dataset split ('train' or 'valid')
        max_samples: Maximum number of samples to load
        cache_dir: Directory to cache the dataset. If None, uses HF_DATASETS_CACHE 
                   environment variable or default location
        
    Returns:
        Tuple of (dataset, vocab_size)
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import os
    except ImportError:
        raise ImportError(
            "datasets and transformers packages are required. "
            "Install with: pip install datasets transformers"
        )
    
    # Determine cache directory
    if cache_dir is None:
        # Check environment variable first
        cache_dir = os.environ.get('HF_DATASETS_CACHE')
        if cache_dir is None:
            # Use HF_HOME if set
            hf_home = os.environ.get('HF_HOME')
            if hf_home:
                cache_dir = os.path.join(hf_home, 'datasets')
    
    # Load TinyStories dataset
    # The dataset is available at roneneldan/TinyStories
    # Alternative: "roneneldan/TinyStories-33M" or "roneneldan/TinyStories-28M"
    dataset_name = "roneneldan/TinyStories"
    
    print(f"Loading TinyStories dataset from Hugging Face...")
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    # Load or create tokenizer
    # Using GPT-2 tokenizer as a default, but can be customized
    tokenizer_name = "gpt2"
    print(f"Loading tokenizer: {tokenizer_name}")
    
    # Set tokenizer cache directory from environment or use dataset cache
    tokenizer_cache = None
    if cache_dir:
        # Use same base directory for tokenizer cache
        import os
        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            tokenizer_cache = os.path.join(hf_home, 'transformers')
        else:
            # Extract base directory from cache_dir
            if cache_dir.endswith('/datasets'):
                tokenizer_cache = cache_dir.replace('/datasets', '/transformers')
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=tokenizer_cache
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset wrapper
    hf_dataset = HuggingFaceDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column='text',
        max_samples=max_samples
    )
    
    vocab_size = len(tokenizer)
    
    return hf_dataset, vocab_size


def create_data_loader(
    cfg,
    split: str = 'train',
    shuffle: bool = True
) -> Tuple[DataLoader, int]:
    """Create a data loader based on configuration.
    
    Args:
        cfg: Configuration object with data settings
        split: Dataset split ('train' or 'valid')
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (data_loader, vocab_size)
    """
    if cfg.data.get('synthetic', True):
        # Create synthetic dataset
        dataset = SyntheticDataset(
            vocab_size=cfg.data.vocab_size,
            seq_len=cfg.data.seq_len,
            num_samples=cfg.data.get('num_samples', 10000)
        )
        vocab_size = cfg.data.vocab_size
        
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )
        
    elif cfg.data.get('dataset_name') == 'tinystories':
        # Load TinyStories dataset
        dataset, vocab_size = load_tinystories_dataset(
            seq_len=cfg.data.seq_len,
            split=split,
            max_samples=cfg.data.get('max_samples'),
            cache_dir=cfg.data.get('cache_dir')
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )
        
    else:
        raise ValueError(
            f"Unknown data configuration. "
            f"Set synthetic=true or dataset_name=tinystories"
        )
    
    return data_loader, vocab_size

