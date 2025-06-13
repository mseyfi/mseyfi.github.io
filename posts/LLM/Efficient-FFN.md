[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

## Beyond Attention: A Deep Dive into Efficient FFN Architectures in LLMs

While attention mechanisms get much of the spotlight, the Feed-Forward Network (FFN) is the computational workhorse of the Transformer. Typically comprising two-thirds of a model's parameters, the FFN block is a critical target for efficiency innovations. As of mid-2025, the strategies have evolved far beyond simply tweaking dimensions.

This tutorial provides a deep, technical breakdown of the most prominent efficient FFN architectures, analyzing their mathematical foundations, performance trade-offs, and practical implementations.

---
### The Baseline: The Standard Transformer FFN

This is the vanilla FFN architecture introduced in "Attention Is All You Need" and used in many early models.

* **Intuition:** The FFN's role is often described as a "memory" or "knowledge" layer. A more functional intuition is that of a **feature expander and refiner** 💡. The up-projection layer takes token representations and projects them into a much higher-dimensional space. In this expanded space, the model has more "room" to identify, separate, and amplify complex combinations of features. The down-projection then integrates this richer feature information back into the original dimension, providing a more refined representation to the next layer.

* **Mathematical Formulation:**
    Let the input be $x \in \mathbb{R}^{B \times L \times d_{model}}$, where $B$ is batch size, $L$ is sequence length. The FFN has an intermediate dimension $d_{ff}$ (typically $4 \times d_{model}$).
    1.  **Up-projection:** A linear layer with weights $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and biases $b_1 \in \mathbb{R}^{d_{ff}}$.
        $$H = xW_1 + b_1$$
        The resulting hidden representation $H$ has the shape $\mathbb{R}^{B \times L \times d_{ff}}$.
    2.  **Activation:** A non-linear function like GELU is applied element-wise.
        $$H_{act} = \text{GELU}(H)$$
    3.  **Down-projection:** Another linear layer with weights $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ and biases $b_2 \in \mathbb{R}^{d_{model}}$.
        $$y = H_{act}W_2 + b_2$$
        The final output $y$ has the shape $\mathbb{R}^{B \times L \times d_{model}}$.

* **Complexity Analysis:**
    * **Parameter Count:** The total number of learnable parameters is the sum of weights and biases from both layers.
        $$P_{FFN} = (d_{model} \cdot d_{ff} + d_{ff}) + (d_{ff} \cdot d_{model} + d_{model}) = 2 \cdot d_{model} \cdot d_{ff} + d_{ff} + d_{model}$$
        For $d_{ff} = 4d_{model}$, this is approximately $8 \cdot d_{model}^2$.
    * **Computational Complexity (FLOPs):** We approximate the FLOPs for a forward pass by counting the multiply-add operations in the linear layers (approximately $2 \times \text{input\_dim} \times \text{output\_dim}$).
        * Up-projection FLOPs: $B \cdot L \cdot (2 \cdot d_{model} \cdot d_{ff})$
        * Down-projection FLOPs: $B \cdot L \cdot (2 \cdot d_{ff} \cdot d_{model})$
        * **Total FLOPs:** $\approx B \cdot L \cdot 4 \cdot d_{model} \cdot d_{ff}$. For $d_{ff} = 4d_{model}$, this is $\approx 16 \cdot B \cdot L \cdot d_{model}^2$.

* **Code Snippet:**
    ```python
    import torch.nn as nn

    class StandardFFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.activation = nn.GELU()
            self.linear2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
    ```

* **Key Reference:**
    * Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762.

---
### 1. Mixture of Experts (MoE)

MoE fundamentally changes the scaling law of Transformers by introducing conditional computation.

* **Intuition:** The specialist analogy is key 🧑‍⚕️. A single, monolithic FFN must be a generalist. MoE posits that it's more efficient to have a large team of "specialists" (expert FFNs), each skilled at handling different types of information. A lightweight "router" acts as a manager, quickly examining each incoming token and dispatching it to the two most relevant specialists. This allows the model's total knowledge (parameter count) to be vast, while the amount of work done for any single token remains small and fixed.

* **Mathematical Formulation:**
    Let there be $N$ experts, where each expert $E_i$ is a standard FFN. Let $k$ be the number of experts to activate (e.g., k=2).
    1.  **Routing:** An input token representation $x \in \mathbb{R}^{d_{model}}$ is passed through a gating network (a linear layer $W_g \in \mathbb{R}^{d_{model} \times N}$) to produce logits.
        $$\text{logits} = x W_g$$
    2.  **Gate Values & Selection:** A softmax is applied to the logits. The 'Top-K` function selects the indices and weights of the $k$ most likely experts.
        $$G = \text{Softmax}(\text{logits})$$     $$(\text{indices}, \text{gates}) = \text{TopK}(G, k)$$
    3.  **Conditional Expert Processing:** The input token $x$ is processed *only* by the selected experts.
        $$\text{expert\_outputs}_i = E_i(x) \quad \text{for } i \in \text{indices}$$
    4.  **Weighted Combination:** The final output is the sum of the expert outputs, weighted by their corresponding gate values.
        $$y = \sum_{i \in \text{indices}} \text{gates}_i \cdot \text{expert\_outputs}_i$$

* **Complexity Analysis:**
    * **Parameter Count:** The parameters are the sum of the router and all $N$ experts.
        $$P_{MoE} = (d_{model} \cdot N) + (N \cdot P_{FFN})$$
        This is enormous. For a model with 64 experts, the FFN parameters are roughly 64 times larger than the baseline.
    * **Computational Complexity (FLOPs):** This is the magic of MoE. The computation is decoupled from the total number of experts, $N$.
        * Router FLOPs: $B \cdot L \cdot (2 \cdot d_{model} \cdot N)$
        * Expert FLOPs: $B \cdot L \cdot k \cdot (4 \cdot d_{model} \cdot d_{ff})$  (Each token is processed by $k$ experts)
        * **Total FLOPs:** $\approx B \cdot L \cdot (2 \cdot d_{model} \cdot N + 4 \cdot k \cdot d_{model} \cdot d_{ff})$. The expert computation dominates. Crucially, the FLOPs scale with $k$ (e.g., 2), not $N$ (e.g., 64).

* **Code Snippet:**
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MoE_FFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int, n_experts: int, k: int):
            super().__init__()
            self.k = k
            self.d_model = d_model
            self.gate = nn.Linear(d_model, n_experts) # Gating network
            self.experts = nn.ModuleList([StandardFFN(d_model, d_ff) for _ in range(n_experts)])

        def forward(self, x):
            B, L, _ = x.shape
            x_flat = x.view(-1, self.d_model)
            
            router_logits = self.gate(x_flat)
            gates, indices = torch.topk(F.softmax(router_logits, dim=-1), self.k, dim=-1)
            gates = gates / torch.sum(gates, dim=-1, keepdim=True)
            
            output = torch.zeros_like(x_flat)
            
            # This is a simplified, non-optimized implementation for clarity
            for i, expert in enumerate(self.experts):
                token_indices, _ = torch.where(indices == i)
                if token_indices.numel() > 0:
                    expert_gates = gates[token_indices, indices[token_indices] == i]
                    expert_output = expert(x_flat[token_indices])
                    output.index_add_(0, token_indices, expert_output * expert_gates.unsqueeze(-1))
            
            return output.view(B, L, -1)
    ```

* **Key References:**
    * Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv preprint arXiv:1701.06538.
    * Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv preprint arXiv:2101.03961.

---
### 2. Gated Linear Units (SwiGLU)

This architecture improves the effectiveness of the core FFN block itself, becoming the default for most modern dense models.

* **Intuition:** The standard FFN is a blunt instrument. A Gated Linear Unit introduces a **dynamic filter** 밸브. Imagine two parallel pipes: one carries the main "content" signal, and the other carries a "control" signal (the gate). The control signal, for each individual feature, determines how much of the content signal is important and should be let through. This allows the network to suppress irrelevant information and amplify important signals with much finer granularity.

* **Mathematical Formulation:**
    Let the input be $x \in \mathbb{R}^{B \times L \times d_{model}}$. SwiGLU uses three weight matrices: $W_1, W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$.
    1.  **Value Path:** The input is projected to create the "value."
        $$V = xW_1$$
    2.  **Gate Path:** In parallel, the input is projected to create the "gate."
        $$G = xW_2$$
    3.  **Gating Mechanism:** The gate is passed through a Swish (or SiLU) activation and multiplied element-wise ($\odot$) with the value.
        $$H = \text{SiLU}(G) \odot V$$
    4.  **Down-projection:** The gated representation is projected back down.
        $$y = HW_3$$

* **Complexity Analysis:**
    * **Parameter Count:** Sum of the three weight matrices (ignoring biases).
        $$P_{SwiGLU} = (d_{model} \cdot d_{ff}) + (d_{model} \cdot d_{ff}) + (d_{ff} \cdot d_{model})$$
        To keep the parameter count comparable to a standard FFN's $2 \cdot d_{model} \cdot d_{ff}$, the intermediate dimension $d_{ff}$ is often set to about $\frac{2}{3}$ of its original size (e.g., $d_{ff} \approx \frac{2}{3} \cdot 4 \cdot d_{model}$).
    * **Computational Complexity (FLOPs):**
        * $W_1$ and $W_2$ projections: $B \cdot L \cdot (2 \cdot d_{model} \cdot d_{ff})$ each.
        * $W_3$ down-projection: $B \cdot L \cdot (2 \cdot d_{ff} \cdot d_{model})$
        * **Total FLOPs:** $\approx B \cdot L \cdot (4 \cdot d_{model} \cdot d_{ff} + 2 \cdot d_{ff} \cdot d_{model})$. This is roughly 1.5x the FLOPs of a standard FFN for the same $d_{ff}$. The trade-off is that SwiGLU's superior performance often allows for a smaller $d_{ff}$ or fewer layers, resulting in a better model for the same overall FLOPs budget.

* **Code Snippet:**
    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    class SwiGLUFFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff, bias=False) # The "value"
            self.w2 = nn.Linear(d_model, d_ff, bias=False) # The "gate"
            self.w3 = nn.Linear(d_ff, d_model, bias=False) # The down-projection
            
        def forward(self, x):
            value = self.w1(x)
            gate = self.w2(x)
            gated_value = F.silu(gate) * value
            output = self.w3(gated_value)
            return output
    ```
* **Key Reference:**
    * Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv preprint arXiv:2002.05202.

---
### 3. Cross-Layer Parameter Sharing

This is a direct method for parameter efficiency, trading model capacity for a smaller memory footprint.

* **Intuition:** A model doesn't necessarily need a brand new "brain" at every layer. By forcing it to reuse the same set of parameters, we are essentially making it learn a more general, **iterative refinement function** 🔄. The same function is applied repeatedly to the hidden state, deepening the representation at each step. Think of it less like a 12-stage assembly line with different tools, and more like a single, highly versatile tool that is applied 12 times to shape the final product.

* **Mathematical Formulation:**
    Let `FFN_shared` be a single FFN module and `Attn_l` be the attention module at layer `l`. The update rule for a layer `l` (in a simplified pre-norm architecture) would look like:
    $$h'_{l} = h_{l-1} + \text{Attn}_l(h_{l-1})$$ $$h_{l} = h'_{l} + \text{FFN}_{\text{shared}}(h'_{l})$$
    The key is that the weights within `FFN_shared` are identical for all layers $l=1, \dots, N_{layers}$.

* **Complexity Analysis:**
    * **Parameter Count:** The FFN parameters do **not** scale with the number of layers, $N_{layers}$.
        $$P_{TotalFFN} = P_{FFN} = 2 \cdot d_{model} \cdot d_{ff} + \dots$$
        This is a massive reduction from the baseline's $N_{layers} \cdot (2 \cdot d_{model} \cdot d_{ff} + \dots)$.
    * **Computational Complexity (FLOPs):** The FLOPs for a full forward pass are **unchanged** compared to the baseline. We still have to execute the FFN matrix multiplications at every layer. **The efficiency gain is in parameter storage (memory footprint), not in computational speed during inference.**

* **Code Snippet:** This is an architectural pattern, not a self-contained module. You implement it when building the full Transformer stack.
    ```python
    import torch.nn as nn
    from typing import List

    # This is a placeholder for any attention implementation
    class Attention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.qkv_proj = nn.Linear(d_model, d_model * 3)
            self.out_proj = nn.Linear(d_model, d_model)
    
    class TransformerWithSharedFFN(nn.Module):
        def __init__(self, num_layers: int, d_model: int, d_ff: int):
            super().__init__()
            self.num_layers = num_layers
            # We still need separate attention layers
            self.attention_layers = nn.ModuleList([Attention(d_model) for _ in range(num_layers)])
            # But only one FFN is instantiated and stored
            self.ffn = StandardFFN(d_model, d_ff)
            self.layernorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers * 2)])

        def forward(self, x):
            for i in range(self.num_layers):
                # Apply attention, residual connection, and norm
                x = x + self.attention_layers[i](self.layernorms[2*i](x))
                # Apply the SHARED FFN, residual connection, and norm
                x = x + self.ffn(self.layernorms[2*i + 1](x))
            return x
    ```
* **Key Reference:**
    * Lan, Z., Chen, M., Goodman, S., et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. arXiv preprint arXiv:1909.11942.
