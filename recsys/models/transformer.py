import math

import pandas as pd
import torch
from torch import nn
from recsys.config import device


class TokenDrop(nn.Module):
    """For a batch of tokens indices, randomly replace a non-specical token with <pad>.

    Args:
        prob (float): probability of dropping a token
        pad_token (int): index for the <pad> token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(
        self, prob: float = 0.1, pad_token: int = 0, num_special: int = 4
    ) -> None:
        self.prob = prob
        self.num_special = num_special
        self.pad_token = pad_token

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Randomly sample a bernoulli distribution with p=prob
        # to create a mask where 1 means we will replace that token
        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()

        # only replace if the token is not a special token
        can_drop = (sample >= self.num_special).long()
        mask = mask * can_drop

        replace_with = (self.pad_token * torch.ones_like(sample)).long()

        sample_out = (1 - mask) * sample + mask * replace_with

        return sample_out


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings module."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate sinusoidal positional embeddings
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1))
        )
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and causal masking."""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4) -> None:
        super(TransformerBlock, self).__init__()

        # Layer normalization for input
        self.norm1 = nn.LayerNorm(hidden_size)

        # Multi-head self-attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )

        # Layer normalization for attention output
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feedforward neural network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # Create causal mask for Attention
        bs, l, h = x.shape
        mask = torch.triu(torch.ones(l, l, device=x.device), 1).bool()

        # Layer normalization
        norm_x: torch.Tensor = self.norm1(x)

        # Apply multi-head Attention
        x = (
            self.multihead_attn(
                norm_x, norm_x, norm_x, attn_mask=mask, key_padding_mask=padding_mask
            )[0]
            + x
        )

        # Layer normalization
        norm_x = self.norm2(x)

        # Apply feedforward neural network
        x = self.mlp(norm_x) + x
        return x


class Transformer(nn.Module):
    """'Decoder-Only' Style Transformer with self-attention."""

    def __init__(
        self,
        embedings_df: pd.DataFrame,
        num_emb: int,
        hidden_size: int = 768,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        super(Transformer, self).__init__()

        # Token embeddings
        self.embeddings_tensor = torch.tensor(
            embedings_df["Embed_comb_text"].values.tolist(), device=device
        )
        # Positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # List of Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # Mask for padding tokens
        input_key_mask = input_seq == 0

        h = 768
        bs, l = input_seq.shape

        output_tensor = self.embeddings_tensor[input_seq]

        # Add positional embeddings to token embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb: torch.Tensor = self.pos_emb(seq_indx)
        pos_emb = pos_emb.reshape(1, l, h).expand(bs, l, h)
        embs = output_tensor + pos_emb

        # Pass through Transformer blocks
        for block in self.blocks:
            embs = block(embs, padding_mask=input_key_mask)

        # Output predictions
        return self.fc_out(embs)
