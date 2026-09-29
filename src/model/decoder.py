"""Transformer decoder block and stack.

Masked self-attention + MLP + residual connections + LayerNorm, stacked
N times, with positional embeddings over the combined [visual tokens] +
[letter tokens] sequence.
"""
import torch
from torch import nn
from src.model.attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), #nonlinear transformation
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mask=None):
        normalized = self.ln_1(x)
        attention_result = self.attn(normalized, mask)
        x = x + attention_result
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerDecoder(nn.Module):
    """
    Stacks multiple TransformerBlocks. 
    The tests expect the last class in this file to be the main module[cite: 2].
    """
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)