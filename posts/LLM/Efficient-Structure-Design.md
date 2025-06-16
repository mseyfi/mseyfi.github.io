[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# Efficient Structure Designs for LLMs.

![image](../../images/Efficient-Structure.png)
*Fig.1 Efficient Structure Design for LLMS*


# Efficient FFN
## [![ALBERT](https://img.shields.io/badge/ALBERT-Cross_Layer_Parameter_Sharing:_Efficient_FFN_Structures-blue?style=for-the-badge&logo=github)](../posts/LLM/ALBERT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While innovations like MoE and SwiGLU focus on redesigning the FFN block within a single Transformer layer, Cross-Layer Parameter Sharing tackles efficiency from a different angle: it re-evaluates the relationship between the layers themselves. It asks a simple but profound question: "Does every layer in a deep network truly need its own unique set of weights?"
</div>

## [![MOE](https://img.shields.io/badge/MOE_Mixture_of_Experts:_FFN-Efficient_FFN_Structures-blue?style=for-the-badge&logo=github)](Efficient-FFN)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While attention mechanisms get much of the spotlight, the Feed-Forward Network (FFN) is the computational workhorse of the Transformer. Typically comprising two-thirds of a model's parameters, the FFN block is a critical target for efficiency innovations. As of mid-2025, the strategies have evolved far beyond simply tweaking dimensions.
</div>

## [![SwiGlu](https://img.shields.io/badge/SWiGLU-The_Modern_FFN_for_High_Performance_Transformers-blue?style=for-the-badge&logo=github)](SWiGLU)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While Mixture of Experts (MoE) offers a path to scale models to enormous parameter counts via sparsity, another critical innovation has optimized the performance of the core "dense" Transformer block itself. This is the <b>Gated Linear Unit (GLU)</b>, and specifically its most successful variant, <b>SwiGLU</b>.

As of mid-2025, SwiGLU has replaced the traditional FFN in most state-of-the-art dense language models (like Meta's Llama series, Google's PaLM, and Mistral's dense models). This tutorial explores the intuition, mathematics, and implementation behind this powerful and efficient architecture.
</div>


# Low Complexity Attention
## Low Rank Attention

## [![Linformer](https://img.shields.io/badge/Linformer-A_Primary_Example_of_Explicit_Low_Rank_Attention-blue?style=for-the-badge&logo=github)](Linformer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Overview of Linformer, why we need it, and side-by-side pseudo-code comparing traditional self-attention to Linformer self-attention. We’ll keep the example to single-head attention for clarity, but in practice you would typically use multi-head attention (with separate projections for each head).

</div>

## Kernel Based Attention
## [![Performer](https://img.shields.io/badge/Performer-Linear_Kernel_approximation_of_Attention-blue?style=for-the-badge&logo=github)](Performer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Performers are Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attentionkernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. Performer demonstrates competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.
</div>

# Multi Query Attention & Grouped Query Attention
## [![MQA](https://img.shields.io/badge/MQA_GQA-Multi_Query_Attention,_Grouped_Query_Attention-blue?style=for-the-badge&logo=github)](MQA-GQA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
The Transformer architecture's self-attention mechanism is the engine of modern AI. However, as models and their context windows grow, the computational and memory costs of standard Multi-Head Attention (MHA) become a significant bottleneck, especially during inference.

This tutorial provides an in-depth exploration of two powerful solutions: Multi-Query Attention <b>(MQA)</b> and Grouped-Query Attention <b>(GQA)</b>. Understanding these architectural details, including their mathematical foundations and impact on memory, is essential for grasping how models like Google's Gemini, Meta's Llama 3, and Mistral AI's models operate so efficiently.

</div>

