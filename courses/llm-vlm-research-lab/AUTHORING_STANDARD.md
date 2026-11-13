# LLM · VLM · Video Research Lab — Long-form teaching standard
**Updated:** 2026-11-12. This is a quality specification, not a declaration that every session already meets it.

## Required page format
The reader must be able to scroll from beginning to end without switching Overview/Math/Code/Lab tabs. Keep intuitive explanation, concrete figures, math, code, lab and experimental observations **interwoven** where the ideas arise. Essential content stays visible. Add a chapter anchor navigator, accessible controls, mobile layout, and respect prefers-reduced-motion.

## Analogies and visual pedagogy (mandatory)

- **Multiple analogies per session:** normally two to four, strategically spaced through motivation, tensor operations, training dynamics and inference. Each analogy must map its concrete objects to exact mathematical variables and explain where it breaks.
- **Multiple diagrams per session:** target at least three genuinely different figures — architecture/tensor shapes, state or data flow, and numerical or cost model — with clear axes, units and captions.
- **Interactive animation with truth conditions:** at least one playable, algorithm-specific animated simulation and one parameter-controlled numerical experiment; never change colors/particles without changing the algorithmic computation. Support Pause, Reset and reduced motion.
- **A visual-to-math bridge:** directly below each important figure, show a precise formula, its variables and a small example. Add the Python/PyTorch snippet immediately after the mechanism, not in a separate code tab.
- **Do not label figures research measurements** unless the user actually runs the model and records results. Toy simulations are transparent, unit-tested examples.
- **Design cues by track:** Transformers = Q/K/V token arrows and RoPE angle geometry; training = memory ledgers, gradient accumulation and all-gather timelines; inference = roofline and cache growth; VLM = ViT patch mosaics and projector dims; video = timestamped sampling, temporal grounding, retrieval; diffusion = denoising trajectories and ODE solvers.

## Required teaching depth for each session
1. **Problem and motivation.** Establish prerequisites, what the model is for, what earlier baselines fail to do, and at least one realistic example.
2. **Precise method variants.** Define every technical term. Compare competing algorithms, architecture families, and crucial tradeoffs. Distinguish a tokenizer library from a tokenization algorithm, or a model checkpoint from a model family.
3. **Figures and meaningful animations.** Diagrams must show the true computation, axes, token/patch shapes, training vs inference and/or model outputs. An interactive element must simulate the named algorithm's operation, not arbitrary "information units."
4. **Mathematical derivations.** Explain every symbol, shape and normalization axis. Show objectives, gradient flow, training procedures and computational/memory complexity when relevant. Work a numerical example.
5. **Actual Python/PyTorch/Hugging Face.** Interleave runnable code with the discussion; include setup, shapes, input/output example, tests, limitations and reproducibility. Toy algorithms must explicitly say they are toy; fabricated benchmark outputs are prohibited.
6. **Research-grade lab.** Hypothesis, baseline, one-variable ablation, explicit evaluation metric, expected effect, observed effect (only if run), hard negatives, error taxonomy, and a follow-up experiment.
7. **Critical evidence.** Primary papers, official documentation, model cards, relevant specific O'Reilly book chapters/pages, and concrete original research articles. Separate book-derived material from newly researched extensions.
8. **Complete review.** Genuine comprehension questions with worked answers, not generic "identify the input" boilerplate.
9. **Optional read-aloud.** Browser speech synthesis may narrate sections when supported. Label it browser TTS; never imply a prerecorded voice track or server inference.

## Prior posts defining the desired standard
- [ControlNet](../../posts/ControlNet.md): ~49k-character long-form explanation including motivations, architecture figures, zero convolutions, objectives, train/infer and code.
- [RoPE](../../posts/LLM/ROPE.md): ~16k characters of intuition, rotation mathematics/proof, PyTorch and contrasts.
- [Stable Diffusion](../../posts/StableDiffusion.md): VAE, U-Net, noise schedules, objectives and implementation.
- [FlashAttention](../../posts/LLM/FlashAttention.md): memory bottlenecks and IO-aware algorithms.
- [LoRA](../../posts/LLM/FineTune/LORA.md): PEFT foundations and implementation details.

## Core reference
*Vision Language Models: Building VLMs with Hugging Face* (Merve Noyan, Miquel Farré, Andrés Marafioti, Orr Zohar; O'Reilly, June 2026). Book chapters: 1 foundations; 2 applications; 3 VLM training; 4 data; 5 post-training; 6 architectures; 7 deployment; 8 Document AI; 9 Video-Language Models; 10 any-to-any models; 11 agents/research. Book assumes prior ML/Python; fill foundational gaps rather than using unexplained terms. For tokenizer specifics use [Hugging Face components](https://huggingface.co/docs/tokenizers/main/en/components), [SentencePiece docs](https://google.github.io/sentencepiece/) and [tiktoken](https://github.com/openai/tiktoken).

## High-priority paper: efficient LLM inference
- **Highest priority:** [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294) (Zhou et al., v3, July 2024). Use its three-level taxonomy (data / model / system) and the prefill/decode bottleneck analysis as the organizing reference for the inference block.
- [Cross-lesson reading map and required labs](EFFICIENT_INFERENCE_READING_MAP.md). Focus Sessions 19–23 first, especially KV cache, IO-aware attention, quantization, speculative decoding, scheduling and serving; also strengthen Sessions 05, 12 and 13.
- Treat the survey as a foundational 2024 paper, not a verified 2026 SOTA benchmark. Verify newer methods and up-to-date official implementations before making ranking claims.

## Long-form expansion coverage for all 46 sessions
- **01–08:** tokenization and embeddings; autoregressive LM; transformer blocks; scaled attention; MHA/MQA/GQA; RoPE/ALiBi; RMSNorm/SwiGLU/MoE; sampling.
- **09–18:** training data, scaling, optimization, checkpointing, FSDP/ZeRO, TP/PP/CP, SFT, LoRA/QLoRA, reward modeling, DPO/PPO, distillation.
- **19–23:** KV cache, FlashAttention, PagedAttention, speculative decoding, quantization and benchmarking.
- **24–31:** ViT, CLIP/SigLIP, VLM connectors, Q-Former and resamplers, multimodal training and SFT, high-resolution OCR and document AI, grounding.
- **32–39:** video tokens, temporal sampling and encoders, retrieval, temporal grounding, VideoRAG, indexing, motion reasoning and benchmarks.
- **40–44:** DDPM, sampling solvers, latent diffusion, DiT, flow matching, consistency and video generation.
- **45–46:** agents, omni systems, research capstone building real video search and QA.

## Actual state
- **Session 01:** Rewritten as a full-length continuous HTML chapter (~50k characters versus ~11.5k originally), with correct illustrative BPE, tokenizer variants, a numerical embedding diagram, next-token equations, four code labs, topic-specific controls, Q&A and browser TTS. It is a starting quality benchmark, not an assertion of exhaustive SOTA coverage.
- **Sessions 02–46:** UI changed to continuous reading, all five prior sections exposed; optional section TTS. **Original thin content remains** and is still pending rigorous expansion, verification, and replacement of generic demonstrations.
- Only label an entire session deep-complete after verifying diagrams, formulas, code, actual model APIs and a meaningful topic-specific lab.
