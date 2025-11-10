"""Main entry point for SeqBench benchmarking."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.rnn_pytorch import RNNModel
from utils.logging import make_loggers
from utils.bench import latency_sweep_cuda


def generate_synthetic_data(vocab_size: int, seq_len: int, batch_size: int, device: str = 'cuda'):
    """Generate synthetic data for training/benchmarking.
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        batch_size: Batch size
        device: Device to generate data on
        
    Returns:
        Tuple of (input_ids, target_ids)
    """
    # Generate random input sequences
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # For simplicity, target is shifted input (next token prediction)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    target_ids[:, -1] = input_ids[:, 0]  # Wrap around last token
    
    return input_ids, target_ids


def train_model(cfg: DictConfig):
    """Train a model based on configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    # Initialize loggers
    loggers = make_loggers(cfg)
    
    # Log configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    for logger in loggers.values():
        logger.log_config(config_dict)
    
    # Create model based on model name
    model_name = cfg.model.name
    vocab_size = cfg.data.vocab_size
    
    if model_name == 'rnn':
        model = RNNModel(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            dropout=cfg.model.dropout,
            device=device
        )
    else:
        raise ValueError(f"Model {model_name} not yet implemented. Only 'rnn' is supported.")
    
    model = model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_dict = {
        'model': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'd_model': cfg.model.d_model,
        'n_layers': cfg.model.n_layers,
        'dropout': cfg.model.dropout,
        'vocab_size': vocab_size,
        'lr': cfg.trainer.lr,
        'max_steps': cfg.trainer.max_steps,
        'batch_size': cfg.data.batch_size,
        'seq_len': cfg.data.seq_len,
    }
    
    for logger in loggers.values():
        logger.log_params(params_dict)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if cfg.trainer.amp and device == 'cuda' else None
    
    # Compile model if requested
    if cfg.trainer.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Training loop
    model.train()
    pbar = tqdm(range(cfg.trainer.max_steps), desc="Training")
    
    for step in pbar:
        # Generate synthetic data
        input_ids, target_ids = generate_synthetic_data(
            vocab_size=vocab_size,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            device=device
        )
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                # Reshape for loss calculation
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = target_ids.view(-1)
                loss = criterion(logits_flat, targets_flat)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if cfg.trainer.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)
            loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            
            # Gradient clipping
            if cfg.trainer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
            
            optimizer.step()
        
        # Log metrics
        metrics = {'loss': loss.item()}
        for logger in loggers.values():
            logger.log_metrics(metrics, step=step)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Close MLflow run if it exists
    if 'mlflow' in loggers:
        loggers['mlflow'].end_run()
    
    print(f"Training completed. Final loss: {loss.item():.4f}")


def run_latency_benchmark(cfg: DictConfig):
    """Run latency benchmark based on configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    # Initialize loggers
    loggers = make_loggers(cfg)
    
    # Log configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    for logger in loggers.values():
        logger.log_config(config_dict)
    
    # Create model based on model name
    model_name = cfg.model.name
    vocab_size = cfg.data.vocab_size
    
    if model_name == 'rnn':
        model = RNNModel(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            dropout=cfg.model.dropout,
            device=device
        )
    else:
        raise ValueError(f"Model {model_name} not yet implemented. Only 'rnn' is supported.")
    
    model = model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    params_dict = {
        'model': model_name,
        'total_params': total_params,
        'd_model': cfg.model.d_model,
        'n_layers': cfg.model.n_layers,
    }
    
    for logger in loggers.values():
        logger.log_params(params_dict)
    
    # Run latency sweep
    print(f"Running latency benchmark for {model_name}...")
    results = latency_sweep_cuda(
        model=model,
        vocab_size=vocab_size,
        batch_size=cfg.data.batch_size,
        seq_lens=cfg.bench.seq_lens,
        warmup=cfg.bench.warmup,
        iters=cfg.bench.iters,
        tokens_for_latency=cfg.bench.tokens_for_latency,
        device=device
    )
    
    # Print results
    print("\nLatency Results (ms per token):")
    print("-" * 40)
    for seq_len in sorted(results.keys()):
        ms_per_token = results[seq_len]
        print(f"Seq Len {seq_len:4d}: {ms_per_token:.6f} ms/token")
    
    # Log results
    metrics = {f'latency_seq_{seq_len}': ms_per_token for seq_len, ms_per_token in results.items()}
    for logger in loggers.values():
        logger.log_metrics(metrics)
    
    # Close MLflow run if it exists
    if 'mlflow' in loggers:
        loggers['mlflow'].end_run()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function with Hydra decorator.
    
    Args:
        cfg: Hydra configuration object
    """
    # Determine mode (default to train if not specified)
    mode = cfg.get('mode', 'train')
    
    if mode == 'train':
        train_model(cfg)
    elif mode == 'latency':
        run_latency_benchmark(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'latency'.")


if __name__ == "__main__":
    main()

