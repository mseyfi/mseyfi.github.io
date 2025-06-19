[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# Efficient Inference Engine
![inference](../../images/Efficient-Inference.png)


## [![FlashAttention](https://img.shields.io/badge/FLASHATTENTION-Fast_and_Memory_Efficient_Exact_Attention_with_IO_Awareness-blue?style=for-the-badge&logo=github)](FlashAttention)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
FlashAttention is a pivotal innovation that addresses one of the most fundamental bottlenecks in the Transformer architecture. Unlike architectural changes like MoE or GQA, FlashAttention is a groundbreaking implementation that doesn't change the mathematics of attention but radically alters how it's computed on the hardware.
</div>


## [![Spec](https://img.shields.io/badge/Speculative_Decoding-Speculative_Decoding_For_Fast_Inference-blue?style=for-the-badge&logo=github)](Speculative-Decoding)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Even with highly optimized systems, the process of generating text from Large Language Models (LLMs) faces a final, stubborn bottleneck: <b>sequential decoding</b>. Generating one token requires a full, time-consuming forward pass through the model. Because this must be done one token at a time, the overall speed is limited by memory latency.
</div>
