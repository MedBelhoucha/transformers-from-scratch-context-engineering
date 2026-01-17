from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import (
    TransformerConfig, TokenEmbedding, PositionalEncoding,
    TransformerEncoder, TransformerDecoder,
    get_causal_mask, get_key_padding_mask, shift_right
)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, bos_id: int = 1, eos_id: int = 2):
        super().__init__()
        self.cfg = cfg
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.src_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.tgt_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)

        self.encoder = TransformerEncoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)
        self.decoder = TransformerDecoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)

        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        device = src.device
        src_kpm = get_key_padding_mask(src, pad_id=self.cfg.pad_id)
        tgt_kpm = get_key_padding_mask(tgt_inp, pad_id=self.cfg.pad_id)
        causal = get_causal_mask(tgt_inp.size(1), device=device)

        enc = self.src_emb(src)
        enc = enc + self.pos_enc(enc)
        enc = F.dropout(enc, p=self.cfg.dropout, training=self.training)
        enc = self.encoder(enc, key_padding_mask=src_kpm)

        dec = self.tgt_emb(tgt_inp)
        dec = dec + self.pos_enc(dec)
        dec = F.dropout(dec, p=self.cfg.dropout, training=self.training)

        dec = self.decoder(dec, enc_out=enc, self_additive_mask=causal, self_key_padding_mask=tgt_kpm, enc_key_padding_mask=src_kpm)
        return self.out_proj(dec)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 32) -> torch.Tensor:
        self.eval()
        B = src.size(0)
        device = src.device
        src_kpm = get_key_padding_mask(src, pad_id=self.cfg.pad_id)

        enc = self.src_emb(src)
        enc = enc + self.pos_enc(enc)
        enc = self.encoder(enc, key_padding_mask=src_kpm)

        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            causal = get_causal_mask(ys.size(1), device=device)
            tgt_kpm = get_key_padding_mask(ys, pad_id=self.cfg.pad_id)

            dec = self.tgt_emb(ys)
            dec = dec + self.pos_enc(dec)
            dec = self.decoder(dec, enc_out=enc, self_additive_mask=causal, self_key_padding_mask=tgt_kpm, enc_key_padding_mask=src_kpm)

            logits = self.out_proj(dec)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)

            if torch.all(next_tok.squeeze(1) == self.eos_id):
                break

        return ys


def demo():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=150, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=64, dropout=0.1, pad_id=0)
    model = EncoderDecoderTransformer(cfg, bos_id=1, eos_id=2)

    B, S, T = 3, 12, 10
    src = torch.randint(3, cfg.vocab_size, (B, S))
    src[0, -2:] = 0

    tgt = torch.randint(3, cfg.vocab_size, (B, T))
    tgt[1, -3:] = 0
    tgt_inp = shift_right(tgt, bos_id=1)

    logits = model(src, tgt_inp)
    print("src:", src.shape)
    print("tgt_inp:", tgt_inp.shape)
    print("logits:", logits.shape)

    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=cfg.pad_id)
    loss.backward()
    print("loss:", loss.item())

    gen = model.greedy_decode(src, max_len=16)
    print("greedy decoded:", gen.tolist())


if __name__ == "__main__":
    demo()
