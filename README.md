# Transformers from scratch (PyTorch)

## Goal
Implement Transformers **from scratch** in Python using PyTorch primitives only.

### Constraints
Forbidden:
- `torch.nn.Transformer`
- `torch.nn.TransformerEncoder` / `torch.nn.TransformerDecoder`
- `torch.nn.MultiheadAttention`

Allowed:
- PyTorch primitives like `nn.Linear`, `matmul`, `softmax`, `dropout`, `Embedding`, etc.

## Files
- `core.py` : LayerNorm (custom), PositionalEncoding, Scaled Dot-Product Attention, Multi-Head Attention (custom),
  FFN, Encoder/Decoder layers (Pre-Norm), causal + padding masks.
- `encoder_only_classifier.py` : Encoder-only (BERT-like) classifier using [CLS] pooling.
- `decoder_only_lm.py` : Decoder-only (GPT-like) causal LM + `generate()` (greedy).
- `encoder_decoder_seq2seq.py` : Encoder-Decoder (seq2seq) + `greedy_decode()`.

## Run (Colab or Local)
```bash
python encoder_only_classifier.py
python decoder_only_lm.py
python encoder_decoder_seq2seq.py

## Reproducibility
The prompt/context used to generate the code is available in **PROMPT.md**.

