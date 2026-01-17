from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import TransformerConfig, TokenEmbedding, PositionalEncoding, TransformerEncoder, get_key_padding_mask


class EncoderOnlyTransformerClassifier(nn.Module):
    def __init__(self, cfg: TransformerConfig, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.encoder = TransformerEncoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        kpm = get_key_padding_mask(tokens, pad_id=self.cfg.pad_id)  # (B,1,1,T)
        x = self.tok_emb(tokens)
        x = x + self.pos_enc(x)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        x = self.encoder(x, key_padding_mask=kpm)
        cls_state = x[:, 0, :]
        return self.classifier(cls_state)


def demo():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=100, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=64, dropout=0.1, pad_id=0)
    model = EncoderOnlyTransformerClassifier(cfg, n_classes=3)

    B, T = 4, 16
    tokens = torch.randint(2, cfg.vocab_size, (B, T))
    tokens[:, 0] = 1
    tokens[0, -3:] = 0

    logits = model(tokens)
    print("tokens:", tokens.shape)
    print("logits:", logits.shape)

    labels = torch.randint(0, 3, (B,))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print("loss:", loss.item())


if __name__ == "__main__":
    demo()
