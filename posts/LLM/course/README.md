# Chapter 1 — Transformer Architecture: From Attention to Modern LLMs

**Audience:** ML / computer vision research scientists. **Language:** English. **Format:** intuition → derivation → visualization → PyTorch → experiments.

## Learning objectives
By the end of this chapter, you will derive scaled dot-product attention, implement multi-head causal attention, explain normalization and residual pathways, and analyze inference complexity.

## Lessons
1. [Self-Attention: Q, K, V and scaled dot products](01-self-attention.md)
2. Multi-head attention and representation subspaces (planned)
3. Positional information: sinusoidal embeddings, RoPE, ALiBi (planned)
4. Causal masks and decoder-only Transformers (planned)
5. Residual connections, RMSNorm, SwiGLU (planned)
6. MHA vs MQA vs GQA, KV cache and decode latency (planned)
7. FlashAttention and IO-aware computation (planned)
8. Transformer design lab: build a small GPT-style model (planned)

## Research questions
- Why divide attention logits by sqrt(d_k)?
- Why is a causal mask essential for next-token training?
- What is the memory complexity of full attention, and what changes with FlashAttention?
- When does GQA improve throughput, and what representational tradeoff does it introduce?

## Prerequisites
Matrix multiplication, softmax, gradients, PyTorch tensors. Familiarity with Vision Transformers is helpful but not required.

[Back to existing LLM study plan](../LLMMain.md)
