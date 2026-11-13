# Priority research track: Efficient Inference for LLMs

**Priority:** Highest — core to the LLM/VLM research curriculum and technical interview preparation.

**Anchor survey:** Zixuan Zhou et al., *A Survey on Efficient Inference for Large Language Models*, arXiv:2404.14294v3 (19 July 2024). [PDF](https://arxiv.org/pdf/2404.14294) · [Abstract](https://arxiv.org/abs/2404.14294)

**Scope caution:** This is a 2024 survey. It gives a valuable organizing taxonomy; its method comparisons and benchmarks must not be presented as a current 2026 state of the art. Validate newer kernels, systems and model families against their latest primary sources.

## Guiding question

Given the same model family, workload and hardware budget, how can we reduce **time-to-first-token (TTFT), inter-token latency (ITL or TPOT), p95 end-to-end latency, GPU memory, energy, and total serving cost**, while preserving model quality and fairness between requests? Distinguish optimization for prefill, decode, single-request latency, and multi-user throughput.

## Paper taxonomy and lesson mapping

| Axis / survey topic | What to deeply teach | Existing lesson(s) | Priority |
|---|---|---|---|
| Root bottlenecks / Roofline | FLOPs vs HBM bytes, arithmetic intensity, prefill vs decode, bandwidth-vs-compute-bound phases | 03, 04, 19, 22 | Critical |
| Data level: input compression | Prompt pruning/summarization, soft prompts, retrieval-augmented context selection, fidelity–compression tradeoffs | 19, 22, 23 (expand) | High |
| Data level: output organization | Parallelizable output formats and inference DAGs vs inherently sequential autoregressive generation | 02, 21, 22 (expand) | High |
| Model: attention structure | MHA/MQA/GQA, sparse / low-complexity attention, and how KV heads determine memory | 04, 05, 06, 20 | Critical |
| Model: alternative architectures | State-space/linear-attention/hybrid architectures, including where comparisons depend on sequence length | 03, 07 (expand) | High |
| Model compression | Weight-only/weight+activation quantization, PTQ/QAT, calibration/outliers; structured pruning, distillation, low-rank factorization | 15, 18, 22 (expand) | Critical |
| Inference engine | IO-aware attention, operator fusion, GEMM tiling, kernel launch overhead, GPU profiling and offloading | 19, 20, 22 | Critical |
| Speculative decoding | Exact distribution-preserving acceptance/rejection, draft/target economics, output token verification | 21 | Critical |
| Serving and scheduling | Paged KV, prefix caching, continuous batching, chunked prefill, admission/scheduling and fairness | 19, 20, 22 | Critical |
| Distributed serving | Tensor/pipeline parallelism, prefill–decode disaggregation and communication/latency tradeoffs | 12, 13, 22 | High |
| Evaluation | TTFT, TPOT/ITL, p50/p95, throughput, goodput at an SLO, peak memory, quality at fixed resources | 22, 23 | Critical |

## Required worked formulas (with assumptions and tensor shapes)

1. **Weight memory** (ignore scales/metadata/fragmentation): `M_weights = P * b_w / 8` bytes for `P` parameters at `b_w` bits each. Distinguish advertised bit width from actual allocation and kernel support.
2. **KV-cache memory** (uniform-layer decoder; no quantization metadata): `M_KV = 2 * B * L * N_layers * N_KV_heads * d_head * b_kv/8` bytes. Show GQA/MQA reduction relative to MHA, and why scheduling determines effective `B` and `L`.
3. **Attention operation costs:** distinguish naive prefill attention computation roughly `O(L^2 d)` from decode-time per-new-token attention `O(L d)` (not counting projections/FFN and implementation-specific kernels).
4. **Roofline:** performance bounded by `min(Peak_FLOP_per_s, Memory_Bandwidth_bytes_per_s * Arithmetic_Intensity_FLOP_per_byte)`. Derive arithmetic intensity for a representative matrix-vector decode and for batched matrix-matrix prefill.
5. **Speculative acceptance:** for draft distribution `q` and target `p`, accepting proposal token `x` with probability `min(1, p(x)/q(x))` in the standard exact speculative-sampling construction; rejected tokens require corrected residual sampling. Contrast greedy and stochastic verification.
6. **Latency decomposition:** `T_total = T_queue + T_prefill + T_decode + T_other`, reporting the workload definition, prompt/output lengths and batching conditions. Do not confuse token throughput with request throughput or SLO-goodput.

## Essential figures and simulations

- GPU memory-map diagram for weights vs activations vs KV cache across prefill and decode (real dimensions, dynamic batch).
- Interactive KV-cache size calculator varying context length, batch, KV heads, dtype; compare MHA, GQA and MQA.
- Visualization of FlashAttention tiling and HBM↔SRAM traffic, including which intermediate tensors are *not* materialized.
- Animated example of continuous batching with multiple requests arriving and finishing at different times.
- Speculative decoding tree showing proposed tokens, parallel target verification and exact accept/reject sequence.
- Roofline plot showing a decode workload and a prefill workload relative to HBM and compute ceilings.

## Hands-on, reproducible labs

- **Lab A — Bottleneck accounting:** compute weight and KV memory for one published open model and verify GPU peak allocation rather than trusting estimates alone.
- **Lab B — Quantization:** benchmark supported BF16, INT8, and 4-bit paths on a documented model/hardware pair; record perplexity/task quality, prefill tokens/s, decode tokens/s and memory. Do not claim automatic speedups from lower bit width.
- **Lab C — Attention / KV-cache:** measure short vs long context, GQA effects and accuracy/latency tradeoffs with fixed hardware and consistent prompts.
- **Lab D — Serving:** compare static and continuous batching, chunked prefill and concurrency; plot TTFT, ITL, p95, throughput, and goodput.
- **Lab E — Speculation:** implement a small numerical example of exact speculative sampling and compare acceptance and measured wall-clock speedup at fixed output distribution.
- **Lab F — Production tradeoffs:** compare vLLM and an independently configured inference baseline with reproducible versions, prompts, warmup, seeds and profiler traces.

## Integration and editorial workflow

This document is the **cross-lesson master reading map**, not a completed chapter and not a new numbered session. Existing lessons should remain **continuous long-form pages**, weaving intuition, diagrams, formulas, PyTorch and labs together; do not create standalone "math" or "code" tabs. For each mapped lesson, cite both the original survey section and updated primary work or official docs. Keep claims and benchmark numbers versioned by year, GPU and workload.

**Work order:** (1) lesson 19 KV/prefill/decode, (2) 20 FlashAttention/PagedAttention, (3) 22 quantization/serving/benchmarking, (4) 21 speculative decoding, (5) 05 MQA/GQA, (6) 23 evaluation, then expand the supporting model/parallelism chapters. These are priorities, not claims of completion.
