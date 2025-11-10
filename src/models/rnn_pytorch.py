"""RNN model implementation using PyTorch's nn.RNN."""

import torch
import torch.nn as nn
from typing import Optional


class RNNModel(nn.Module):
    """Simple RNN-based sequence model with embedding and output projection."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        """Initialize RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension size
            n_layers: Number of RNN layers
            dropout: Dropout probability
            device: Device to run on
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # RNN
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, d_model)
        
        # Output projection
        logits = self.output_proj(rnn_out)  # (batch_size, seq_len, vocab_size)
        
        return logits

