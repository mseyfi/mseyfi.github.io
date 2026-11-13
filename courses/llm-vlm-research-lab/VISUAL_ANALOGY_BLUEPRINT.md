# Visual & Analogy Blueprint — 14 Technical Tracks

Updated: 2026-11-12. This is a **mandatory content design contract**. It is not a claim that all diagrams exist yet. The original user requirement is extensive *English-only*, accessible explanations, substantial mathematics, modern models, runnable PyTorch, multiple real figures and animations per chapter, and analogy-heavy teaching. All of these must be integrated into the continuous page (not separate Code/Math tabs).

## Per-lesson minimum before calling it complete

- **3–6 distinct figures:** architecture, actual tensor shapes, state or memory movement, numerical worked example and/or failure case. Figures must communicate more than labels in boxes.
- **2–4 analogies:** one for the big picture and others where core math and implementation become difficult. Under every analogy include an explicit mapping and where it breaks down.
- **2 interactive controls:** one animation with Pause/Reset and one interactive mathematical simulator with defined axes, units and nonarbitrary computations.
- **Mathematical development:** define symbols and assumptions; derive equations and show at least one hand-worked example with concrete dimensions.
- **PyTorch/HF examples:** minimal runnable example + more realistic training/inference variant, expected tensor shapes and testable checks.
- **Critical experiment:** baseline, fixed budget, controlled ablation, expected/observed distinction, hard negatives, relevant metrics.
- **Read-aloud:** optional neural TTS with fallback and the ability to pause; NEVER promise every voice works in every browser or embed secret APIs.
- **No artificial completion claims:** all chapters start as unreviewed until content and controls are verified.

## Track-by-track teaching plan

### 01. Transformer & LLM foundations

**Analogies:** Tokens are LEGO bricks; attention is a dynamic reference index; RoPE is two clock hands rotating at different frequencies; MoE resembles specialist departments.

**Figures & meaningful animations:** Byte-level/BPE merge animation with evolving vocabulary; text→IDs→embedding lookup matrix; interactive attention QK^T heatmap with masking; RoPE circular geometry with positions; MoE token-to-expert routing with capacity/auxiliary loss.

**Editorial note:** Session 01 BPE stepper is the benchmark, sessions 02–08 are drafts.

### 02. Data preparation & pretraining

**Analogies:** Curating data resembles sorting ingredients before cooking; near-duplicate data is repeated photocopies; sequence packing is filling bus seats, but boundaries matter.

**Figures & meaningful animations:** MinHash shingles and collision grid; dataset mixture allocation with normalized weights; packing rectangles with attention/loss masks; token budget vs scaling curve.

**Editorial note:** Require real dataset sources and documented filtering tradeoffs.

### 03. Training mathematics & memory

**Analogies:** Adam's moments are long-term ledgers; gradient accumulation is tallying votes before a single decision; checkpointing is recomputing spreadsheet cells.

**Figures & meaningful animations:** Per-parameter memory ledger; animated microbatch gradient accumulation; forward/backward activation retention; precision conversions and overflow examples.

**Editorial note:** New Session 51 provides memory and accumulation demos; more is pending.

### 04. Distributed training systems

**Analogies:** FSDP is splitting a very large book across libraries and gathering pages when needed; pipeline parallelism is an assembly line; all-reduce is a shared tally.

**Figures & meaningful animations:** Rank-wise tensor partition dimensions; time-stepped all-gather/reduce-scatter; GPipe/1F1B pipeline bubbles and microbatch Gantt; bandwidth/latency/compute overlap plot.

**Editorial note:** Show real PyTorch FSDP2/torch.distributed collectives and cost models.

### 05. SFT, LoRA and adapters

**Analogies:** Instruction tuning is apprenticeship using worked examples; LoRA is a small adjustable correction to a fixed master blueprint; QLoRA stores the master compressed.

**Figures & meaningful animations:** Chat template and masked loss heatmap; delta-W=BA factor shapes with rank slider; quantization group/scaling diagrams; comparison of trainable and frozen parameters.

**Editorial note:** Demonstrate actual gradient flow in PEFT and verify model APIs.

### 06. Preference and RL post-training

**Analogies:** Preference feedback is comparing two candidate essays; GRPO uses a group of students as a comparison baseline; reward hacking is gaming an exam rubric.

**Figures & meaningful animations:** Preference-pair loss and log-ratio plot; clipped PPO objective; GRPO group-normalized advantages with adjustable rewards; rollout/compute/reward loops with policy version tracking.

**Editorial note:** Show DPO, PPO, GRPO and RLVR as distinct algorithms, not synonyms.

### 07. Inference I: attention and caching

**Analogies:** KV cache is reusable scratch work; FlashAttention is tiled computation that avoids writing massive intermediate sheets; speculative decoding is a junior draft reviewed by a careful senior.

**Figures & meaningful animations:** Prefill vs decode memory traffic; Q/K/V SRAM-HBM tiled movement; exact KV tensor layout and context growth; speculative accept/reject token tree; roofline plot.

**Editorial note:** New Session 64 has roofline and KV visualization; other sections need dedicated technical simulations.

### 08. Inference II: serving and quantization

**Analogies:** Continuous batching is an elevator taking riders who arrive and leave at different floors; quantization is measuring with a coarser ruler; prefill/decode disaggregation resembles separate specialized factory lines.

**Figures & meaningful animations:** Quantization levels and outlier residuals; requests-versus-time scheduling animation; prefill/decode latency bars with queue events; fair vs unfair scheduling counterexamples; p95 latency and SLO goodput plot.

**Editorial note:** Never claim a quantization path increases speed without checking supported GPU kernels.

### 09. VLM architectures

**Analogies:** ViT patches are a mosaic; a vision-language projector is a learned translator between two vector languages; Q-Former resembles a team of interviewers asking questions of visual evidence.

**Figures & meaningful animations:** Actual patch-grid sliders including aspect ratio and token count; vision feature [B,N,Dv]→projector→[B,N,Dllm]; cross-attention query/key/value dimensions; CLIP logits matrix with positive diagonal; visual token compression.

**Editorial note:** Align to the O'Reilly VLM book and recent model cards.

### 10. Multimodal training and applications

**Analogies:** Image-text alignment resembles matching captions to images; OCR is a magnifying glass, but document layout requires geometry; grounding is placing pins on an image map.

**Figures & meaningful animations:** Contrasting CLIP and SigLIP objectives; response-only loss mask over mixed visual/text tokens; high-resolution crop/tiling overlay; box and phrase grounding with coordinates.

**Editorial note:** Include VQA, OCR, doc AI, visual grounding and evaluation slices.

### 11. Video-language understanding

**Analogies:** Frame sampling is choosing pages from a moving story; motion cues are traffic reports, not guaranteed evidence of meaningful events; long-video memory is a timestamped notebook.

**Figures & meaningful animations:** Storyboard timeline with uniform vs scene/motion sampling; temporal position/channel tokens; spatiotemporal attention diagrams; event order tests with swapped frames; long-video token budget calculator.

**Editorial note:** New Session 77 demonstrates missed events and temporal sampling.

### 12. Video retrieval and grounding

**Analogies:** ANN is a library catalog; reranking is opening shortlisted books; temporal grounding is finding the exact paragraph rather than a book title; fingerprints are identifiers rather than semantic topics.

**Figures & meaningful animations:** Embedding similarity geometry; ANN shortlist vs cross-encoder rerank animation; temporal IoU adjustable start/end rectangles; VideoRAG clips, speech and timestamps; retrieval vs fingerprint transformation robustness.

**Editorial note:** Design labs around evidence, hard negatives and timing.

### 13. Diffusion and generative video

**Analogies:** Diffusion resembles reversing gradually corrupted photography; flow matching is a velocity field directing boats; fast solvers are more accurate navigation steps; consistency distills a long journey into a shortcut.

**Figures & meaningful animations:** Noise-schedule forward path; DDPM/DDIM reverse iteration and scheduler shapes; 2D ODE velocity fields with Euler vs higher order; DiT token flow and conditioning; temporal latent volumes.

**Editorial note:** New Session 84 teaches flow/ODE and solver step error.

### 14. Multimodal agents and capstone

**Analogies:** An agent is a research team: perception observes, planner decides, tools execute, verifier checks; agent memory is an evidence ledger, not a magical perfect recall.

**Figures & meaningful animations:** Perception→plan→tool→observe loop; branching tool decisions and failure recovery; timestamped evidence citations; complete capstone video index pipeline, GPU/latency and retrieval evaluation dashboard.

**Editorial note:** Build one reproducible video search + VideoQA system, not a conceptual architecture alone.

## Review process

Use a chapter checklist: (a) prose readability at non-native English reading level; (b) analogy-to-variable mapping; (c) mathematical dimension checks; (d) interactive numeric checks by hand at boundary values; (e) runnable PyTorch code, deterministic seeds and CPU fallback where possible; (f) official paper/implementation citations; (g) read-aloud disabled when math pronunciation would mislead; (h) keyboard accessibility and reduced-motion preference.

For any large chapter, do not shorten the science to fit a visual card. Keep the *core derivation* visible; make supplementary details expandable when needed. Use original SVG and mathematical simulations instead of copying copyrighted illustrations from papers.
