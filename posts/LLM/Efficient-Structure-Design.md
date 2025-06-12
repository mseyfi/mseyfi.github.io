In this Tutorial we will study efficient structure designs for LLMs.

![image](../../images/Efficient-Structure.png)
*Fig.1 Efficient Structure Design for LLMS*




# Efficient Attention

## Low Rank Attention

## [![Linformer](https://img.shields.io/badge/Linformer-A_Primary_Example_of_Explicit_Low_Rank_Attention-blue?style=for-the-badge&logo=github)](Linformer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Overview of Linformer, why we need it, and side-by-side pseudo-code comparing traditional self-attention to Linformer self-attention. We’ll keep the example to single-head attention for clarity, but in practice you would typically use multi-head attention (with separate projections for each head).

</div>

## Kernel BAsed Methods
## [![Performer](https://img.shields.io/badge/Performer-Linear_Kernel_approximation_of_Attention-blue?style=for-the-badge&logo=github)](Performer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Performers are Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attentionkernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. Performer demonstrates competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.
</div>

