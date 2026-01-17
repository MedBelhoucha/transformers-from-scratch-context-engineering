from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Additive causal mask (1,1,T,T): blocks attention to future positions.
    Values: 0 (allow) / -1e9 (block)
    """
    future = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    mask = mask.masked_fill(future, -1e9)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)


def get_key_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    tokens: (B,T)
    returns bool mask (B,1,1,T) where True means PAD positions (to be masked on keys)
    """
    assert tokens.dim() == 2
    return (tokens == pad_id).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)


class LayerNorm(nn.Module):
    """LayerNorm from scratch (normalize last dim)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (buffer, non-trainable)."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.pe[:, :T, :]


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q,K,V) = softmax(QK^T / sqrt(dk) + masks) V
    """
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (B,H,Tq,Tk)

        if additive_mask is not None:
            scores = scores + additive_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=training)

        out = torch.matmul(attn, v)  # (B,H,Tq,Dh)
        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention from scratch."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attn = ScaledDotProductAttention()
        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(m.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        additive_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, attn = self.attn(
            q, k, v,
            additive_mask=additive_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        out = self._merge_heads(out)
        out = self.w_o(out)
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout
        self.activation = activation
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        return F.gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-Norm Encoder layer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h, additive_mask=None, key_padding_mask=key_padding_mask)
        if self.dropout > 0:
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        x = x + attn_out

        h = self.norm2(x)
        ffn_out = self.ffn(h)
        if self.dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x = x + ffn_out
        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-Norm Decoder layer: masked self-attn + cross-attn + ffn."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.self_mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        self_additive_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        self_out, _ = self.self_mha(h, h, h, additive_mask=self_additive_mask, key_padding_mask=self_key_padding_mask)
        if self.dropout > 0:
            self_out = F.dropout(self_out, p=self.dropout, training=self.training)
        x = x + self_out

        if enc_out is not None:
            h = self.norm2(x)
            cross_out, _ = self.cross_mha(h, enc_out, enc_out, additive_mask=None, key_padding_mask=enc_key_padding_mask)
            if self.dropout > 0:
                cross_out = F.dropout(cross_out, p=self.dropout, training=self.training)
            x = x + cross_out

        h = self.norm3(x)
        ffn_out = self.ffn(h)
        if self.dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x = x + ffn_out
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        self_additive_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                enc_out=enc_out,
                self_additive_mask=self_additive_mask,
                self_key_padding_mask=self_key_padding_mask,
                enc_key_padding_mask=enc_key_padding_mask,
            )
        return x


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    max_len: int = 512
    dropout: float = 0.1
    pad_id: int = 0


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.emb(tokens)


def shift_right(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    B, T = tokens.shape
    out = torch.empty_like(tokens)
    out[:, 0] = bos_id
    out[:, 1:] = tokens[:, :-1]
    return out
