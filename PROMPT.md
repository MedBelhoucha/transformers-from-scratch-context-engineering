# Prompt (Context Engineering) used to generate this project

## Role
You are a senior ML engineer and teacher. Generate clean, runnable Python code in PyTorch implementing Transformers from scratch.

## Hard constraints
Forbidden:
- torch.nn.Transformer
- torch.nn.TransformerEncoder / torch.nn.TransformerDecoder
- torch.nn.MultiheadAttention

Allowed:
- PyTorch primitives (nn.Linear, matmul, softmax, dropout, Embedding, etc.) + custom modules.

## Requirements
- Implement: LayerNorm (custom), sinusoidal PositionalEncoding, scaled dot-product attention, multi-head attention (custom), FFN.
- Architecture: Pre-Norm + residual connections.
- Masks: padding mask + causal mask.
- Provide runnable demos in each file: print shapes + compute loss + loss.backward().

## Deliverables
Generate these files:
1) core.py
2) encoder_only_classifier.py (Encoder-only classifier with [CLS])
3) decoder_only_lm.py (Decoder-only causal LM + generate())
4) encoder_decoder_seq2seq.py (Encoder-Decoder seq2seq + greedy_decode())
