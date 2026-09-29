"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as model(images, input_ids), with
input_ids of shape (B, T), returning logits of shape (B, T_out, 27) where
T_out >= T and the last T logits line up with input_ids.
"""
import torch
from torch import nn
from src.model.encoder import CNNEncoder
from src.model.decoder import TransformerDecoder

class TinyVLM(nn.Module):
    def __init__(self, vocab_size=27, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.encoder = CNNEncoder(out_channels=d_model)
        self.adapter = nn.Linear(d_model, d_model)
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(120, d_model) 
        
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, images, input_ids):
        device = images.device
        
        img_feats = self.encoder(images) 
        B, C, H, W = img_feats.shape
        img_feats = img_feats.view(B, C, -1).transpose(1, 2) 
        visual_tokens = self.adapter(img_feats)
        T_vis = visual_tokens.shape[1]

        text_tokens = self.token_embedding(input_ids)
        
        # Concatenate visual and text sequences
        seq = torch.cat([visual_tokens, text_tokens], dim=1)
        T_total = seq.shape[1]
        
        positions = torch.arange(0, T_total, device=device).unsqueeze(0)
        seq = seq + self.pos_embedding(positions)

        # Causal mask: Visual tokens see each other, text tokens are causal
        mask = torch.tril(torch.ones(T_total, T_total, dtype=torch.bool, device=device))
        mask[:T_vis, :T_vis] = True
        mask = mask.view(1, 1, T_total, T_total)

        seq = self.decoder(seq, mask)
        logits = self.lm_head(seq)
        
        return logits