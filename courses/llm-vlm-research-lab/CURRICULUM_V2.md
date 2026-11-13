# LLM · VLM · Video Research Lab — Survey-aligned curriculum v2

**Created:** 2026-11-12  
**Purpose:** Research-level, multi-session topics (not a fixed 3-month crash course). Original 46 lesson URLs stay unchanged. **Status:** 14 tracks / 88 planned sessions; sessions 02–46 remain concise drafts; new sessions are planned or in active authoring, not silently labeled complete. The standard for every authored page is in `AUTHORING_STANDARD.md`.

## Research tracks and anchor surveys

### 01. Transformer & LLM Foundations
- **Existing sessions:** 1, 2, 3, 4, 5, 6, 7, 8 · **new sessions:** 47
- **Focus:** Tokens, attention, positional encoding, architectures, sampling and efficient alternatives.
- **Survey:** [A Survey of Large Language Models (background survey)](https://arxiv.org/abs/2303.18223)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 02. Data Preparation & LLM Pretraining
- **Existing sessions:** 9 · **new sessions:** 48, 49, 50
- **Focus:** Deduplication, token mixtures, packing, compute-optimal training and stability.
- **Survey:** [Data Preparation for Large Language Models (2026)](https://doi.org/10.1007/s11390-026-5948-8)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 03. Training Mathematics & Memory
- **Existing sessions:** 10, 11 · **new sessions:** 51, 52, 53
- **Focus:** Optimizers, gradients, AMP, FP8, activation checkpointing, profiling, GPU memory accounting.
- **Survey:** [Unifying Data, Memory, and Compute Efficiency in LLM Training (2026)](https://arxiv.org/abs/2606.10706)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 04. Distributed Training Systems
- **Existing sessions:** 12, 13 · **new sessions:** 54, 55, 56, 57
- **Focus:** FSDP2, ZeRO, TP/PP/CP, collective operations, communication overlap and fault tolerance.
- **Survey:** [Efficient Training of LLMs on Distributed Infrastructures (2026)](https://doi.org/10.1007/s44336-026-00038-z)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 05. SFT, LoRA & Efficient Adaptation
- **Existing sessions:** 14, 15 · **new sessions:** 58, 59, 60
- **Focus:** SFT datasets, selective losses, LoRA, QLoRA, adapters and data-efficient post-training.
- **Survey:** [A Survey on Efficient LLM Training: Data-Centric Perspectives (ACL 2025)](https://aclanthology.org/2025.acl-long.1493/)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 06. RL, Preference Learning & Reasoning
- **Existing sessions:** 16, 17, 18 · **new sessions:** 61, 62, 63
- **Focus:** Reward models, DPO, PPO, GRPO, RLVR, preference data and reasoning distillation.
- **Survey:** [Learning from Rewards for Large Language Models (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.970/)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 07. Inference I: Attention, KV Cache & Kernels
- **Existing sessions:** 19, 20, 21 · **new sessions:** 64, 65, 66, 67
- **Focus:** Roofline, prefill/decode, FlashAttention, paged KV, KV compression, speculative decoding.
- **Survey:** [A Survey on Efficient Inference for LLMs (2024)](https://arxiv.org/abs/2404.14294)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 08. Inference II: Quantization, Serving & Evaluation
- **Existing sessions:** 22, 23 · **new sessions:** 68, 69, 70, 71
- **Focus:** Quantization, batching, scheduling, prefill-decode disaggregation, TTFT, TPOT, goodput.
- **Survey:** [I/O for LLM Inference: Storage and Memory Bottlenecks (2026)](https://doi.org/10.1007/s10462-026-11651-1)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 09. Vision–Language Architectures
- **Existing sessions:** 24, 25, 26, 27 · **new sessions:** 72, 73
- **Focus:** ViT, contrastive encoders, projectors, Q-Former, unified tokens and vision compression.
- **Survey:** [Towards Unified Multimodal LLMs (ACL 2026)](https://aclanthology.org/2026.findings-acl.1853/)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 10. Multimodal Training & Applications
- **Existing sessions:** 28, 29, 30, 31 · **new sessions:** 74, 75, 76
- **Focus:** Cross-modal objectives, VLM SFT, OCR, grounded QA and fine-grained localization.
- **Survey:** [Efficient Multimodal Large Language Models (2025)](https://doi.org/10.1007/s44267-025-00099-6)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 11. Video-Language Understanding
- **Existing sessions:** 32, 33, 38, 39 · **new sessions:** 77, 78, 79
- **Focus:** Frame/token budgets, time encoding, motion, long video, evaluation and temporal reasoning.
- **Survey:** [Video Understanding with Large Language Models (IEEE, 2025)](https://doi.org/10.1109/TCSVT.2025.3566695)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 12. Video Search, Retrieval & Temporal Grounding
- **Existing sessions:** 34, 35, 36, 37 · **new sessions:** 80, 81, 82, 83
- **Focus:** Video embeddings, ANN indexing, temporal localization, VideoRAG, fingerprinting.
- **Survey:** [Video Temporal Grounding with MLLMs (2025)](https://arxiv.org/abs/2508.10922)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 13. Diffusion, Flow & Generative Video
- **Existing sessions:** 40, 41, 42, 43, 44 · **new sessions:** 84, 85, 86
- **Focus:** Noise prediction, samplers, latent diffusion, DiT, flow matching, distillation and fast generation.
- **Survey:** [Survey of Video Diffusion Models (2025)](https://arxiv.org/abs/2504.16081)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

### 14. Multimodal Agents & Capstone
- **Existing sessions:** 45, 46 · **new sessions:** 87, 88
- **Focus:** Perception/planning/tool use, video QA systems, production evaluation and capstone.
- **Survey:** [Large Multimodal Agents: A Survey (2025)](https://doi.org/10.1007/s44267-025-00093-y)
- **Required deliverables:** intuitive analogy with failure boundary; interactive architecture and tensor-shape diagram; mathematics with derivation; executable PyTorch example; topic-specific simulator; training/inference differences; critical benchmarks; spaced-repetition questions.

## New lesson titles

- Session 47 — Sparse, Linear and Hybrid Attention
- Session 48 — Corpus Filtering, Deduplication and Synthetic Data
- Session 49 — Token Mixtures, Packing and Compute-Optimal Scaling
- Session 50 — Full Pretraining Step: Forward, Loss, Backward and Update
- Session 51 — GPU Memory Anatomy: AdamW, Accumulation and Checkpointing
- Session 52 — BF16, FP8 and Numerical Training Stability
- Session 53 — Activation Recompute, torch.compile and GPU Profiling
- Session 54 — FSDP2, ZeRO and Sharded Optimizer States
- Session 55 — Tensor, Pipeline and Context Parallelism Lab
- Session 56 — Distributed Checkpoints and Fault Recovery
- Session 57 — MFU, Communication Overlap and Bottleneck Analysis
- Session 58 — SFT Data, Chat Templates and Loss Masking
- Session 59 — LoRA Targets, Rank and Initialization
- Session 60 — QLoRA, Quantized Optimizers and Adapter Tradeoffs
- Session 61 — GRPO and Reinforcement Learning with Verifiable Rewards
- Session 62 — Reward Modeling and Reward Hacking
- Session 63 — Multi-Stage Post-Training and Reasoning Distillation
- Session 64 — Roofline Analysis: Prefill versus Decode
- Session 65 — Prefix Caching, Paged KV and KV Layout
- Session 66 — KV Quantization, Compression, Eviction and Offloading
- Session 67 — Exact and Approximate Speculative Decoding
- Session 68 — Continuous Batching and Request Scheduling
- Session 69 — PTQ, QAT, INT8, W4A16 and Cache Quantization
- Session 70 — Prefill–Decode Disaggregation and Parallel Serving
- Session 71 — TTFT, TPOT, Goodput, Latency and Benchmark Design
- Session 72 — Dynamic Image Resolution and Vision Token Compression
- Session 73 — Unified Multimodal Tokens and Fusion Architectures
- Session 74 — Multimodal Pretraining Objectives and Data Mixtures
- Session 75 — Visual SFT, Multimodal RL and Instruction Tuning
- Session 76 — OCR, Document AI and Fine-Grained Grounding
- Session 77 — Long-Video Memory, Sampling and Token Budgets
- Session 78 — Audio-Video Alignment and Temporal Representation Training
- Session 79 — Temporal Reasoning Benchmarks and Counterfactual Evaluation
- Session 80 — Video Embeddings, ANN and Reranking Architecture
- Session 81 — Text-to-Moment Grounding and Localization
- Session 82 — VideoRAG, Evidence Selection and Timestamped Answers
- Session 83 — Video Fingerprinting vs Semantic Video Retrieval
- Session 84 — Flow Matching, Rectified Flow and DiT
- Session 85 — Consistency, Distillation and Few-Step Generation
- Session 86 — Video Generation, Motion Coherence and Evaluation
- Session 87 — Perception, Memory and Tool Use in Multimodal Agents
- Session 88 — Capstone: Build and Benchmark Video Retrieval + QA

## Technical writing policy

1. **Analogy first, then literal computation.** Each analogy must map elements to variables and explicitly state what the analogy *does not* capture.
2. **Figure-rich, but correct.** Diagrams show actual data flow, tensor shapes and training-vs-inference states. Avoid generic boxes and meaningless sliders.
3. **Full derivations and code together.** No isolated math/code tabs. Code blocks belong beside the concept explained.
4. **Topic-specific animated simulators.** Include at least one functional model per session: a token merge, attention heatmap, KV budget, all-gather, policy objective, sampler step, timeline etc.
5. **Voice:** the old browser SpeechSynthesis quality is poor. Provide optional natural neural TTS when technically available; never put paid credentials in public GitHub Pages. Keep accessible text and fallback.
6. **Honest lesson status.** Planned pages have no link until they exist; progress is browser-local and import/exportable.
7. **Survey + primary papers.** Survey defines breadth; verify 2026 primary models, implementation repositories and benchmarks before making state-of-the-art claims.
