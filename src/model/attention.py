"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.

The provided tests expect the last nn.Module defined in this file to be
Cls(d_model, n_heads), called as attention(x, mask) with x of shape
(B, T, d_model). `mask` is optional; when given it is a boolean tensor where
True means "this query may attend to this key".
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Handwritten multi-head attention matching F.scaled_dot_product_attention.
    Test expects: attention = Cls(d_model, n_heads)[cite: 3].
    """
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        H = self.n_heads
        D = C // H


        # Project and split heads: (B, T, C) -> (B, H, T, D)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            # Mask is broadcastable to (B, 1, T, T)
            # True means "may attend", False means block
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)

        
        # Merge heads: (B, H, T, D) -> (B, T, C)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
