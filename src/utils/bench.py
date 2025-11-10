"""Benchmarking utilities for latency measurement using CUDA events."""

import torch
from typing import List, Dict, Any
import numpy as np


def latency_sweep_cuda(
    model: torch.nn.Module,
    vocab_size: int,
    batch_size: int,
    seq_lens: List[int],
    warmup: int = 5,
    iters: int = 20,
    tokens_for_latency: int = 256,
    device: str = 'cuda'
) -> Dict[int, float]:
    """Perform latency sweep using CUDA events.
    
    Args:
        model: PyTorch model to benchmark
        vocab_size: Vocabulary size for input generation
        batch_size: Batch size for benchmarking
        seq_lens: List of sequence lengths to test
        warmup: Number of warmup iterations
        iters: Number of measurement iterations
        tokens_for_latency: Number of tokens to process for latency calculation
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping sequence length to milliseconds per token
    """
    model.eval()
    model = model.to(device)
    
    results = {}
    
    with torch.no_grad():
        for seq_len in seq_lens:
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # Warmup
            for _ in range(warmup):
                _ = model(input_ids)
            
            # Synchronize before timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Create CUDA events for timing
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Measure latency
                start_event.record()
                for _ in range(iters):
                    _ = model(input_ids)
                end_event.record()
                
                # Wait for events to complete
                torch.cuda.synchronize()
                
                # Calculate elapsed time in milliseconds
                elapsed_ms = start_event.elapsed_time(end_event) / iters
                
                # Calculate milliseconds per token
                # We process batch_size * seq_len tokens per forward pass
                tokens_per_forward = batch_size * seq_len
                ms_per_token = elapsed_ms / tokens_per_forward
                
            else:
                # CPU fallback using time.time()
                import time
                start_time = time.time()
                for _ in range(iters):
                    _ = model(input_ids)
                elapsed_seconds = (time.time() - start_time) / iters
                elapsed_ms = elapsed_seconds * 1000
                tokens_per_forward = batch_size * seq_len
                ms_per_token = elapsed_ms / tokens_per_forward
            
            results[seq_len] = ms_per_token
    
    return results

