from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import TransformerConfig, TokenEmbedding, PositionalEncoding, TransformerDecoder, get_causal_mask, get_key_padding_mask


class DecoderOnlyTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.decoder = TransformerDecoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        device = tokens.device
        causal = get_causal_mask(T, device=device)
        kpm = get_key_padding_mask(tokens, pad_id=self.cfg.pad_id)

        x = self.tok_emb(tokens)
        x = x + self.pos_enc(x)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        x = self.decoder(x, enc_out=None, self_additive_mask=causal, self_key_padding_mask=kpm, enc_key_padding_mask=None)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:
        self.eval()
        out = prompt
        for _ in range(max_new_tokens):
            logits = self(out)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
            if out.size(1) >= self.cfg.max_len:
                break
        return out


def demo():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=120, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=64, dropout=0.1, pad_id=0)
    model = DecoderOnlyTransformerLM(cfg)

    B, T = 4, 16
    tokens = torch.randint(1, cfg.vocab_size, (B, T))
    tokens[0, -2:] = 0

    logits = model(tokens)
    print("tokens:", tokens.shape)
    print("logits:", logits.shape)

    inp = tokens[:, :-1]
    tgt = tokens[:, 1:]
    logits_inp = model(inp)
    loss = F.cross_entropy(logits_inp.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=cfg.pad_id)
    loss.backward()
    print("loss:", loss.item())

    prompt = torch.tensor([[5, 6, 7]])
    gen = model.generate(prompt, max_new_tokens=10)
    print("generated:", gen.tolist())


if __name__ == "__main__":
    demo()
