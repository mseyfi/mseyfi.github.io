[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

## Beyond Attention: A Deep Dive into Efficient FFN Architectures in LLMs

While attention mechanisms get much of the spotlight, the Feed-Forward Network (FFN) is the computational workhorse of the Transformer. Typically comprising two-thirds of a model's parameters, the FFN block is a critical target for efficiency innovations. As of mid-2025, the strategies have evolved far beyond simply tweaking dimensions.

This tutorial provides a deep, technical breakdown of the most prominent efficient FFN architectures, analyzing their mathematical foundations, performance trade-offs, and practical implementations.

---
### The Baseline: The Standard Transformer FFN

This is the vanilla FFN architecture introduced in "Attention Is All You Need" and used in many early models.

* **Intuition:** The FFN's role is often described as a "memory" or "knowledge" layer. A more functional intuition is that of a **feature expander and refiner**. The up-projection layer takes token representations and projects them into a much higher-dimensional space. In this expanded space, the model has more "room" to identify, separate, and amplify complex combinations of features. The down-projection then integrates this richer feature information back into the original dimension, providing a more refined representation to the next layer.

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

In the world of Large Language Models (LLMs), there is a constant tension between model size and computational cost. Larger models, with more parameters, tend to be more capable. However, traditional "dense" models become proportionally more expensive to run as they grow—every single parameter is used to process every single token.

As of mid-2025, the **Mixture of Experts (MoE)** architecture has emerged as the dominant solution to this scaling dilemma. It allows for the creation of models with extraordinarily high parameter counts while keeping the computational cost for inference surprisingly low. This guide breaks down exactly how it works.

### 1. The Intuition: A Team of Specialists

To build a strong mental model, let's use an analogy.

* **A Dense Model is a Star Generalist Doctor:** Imagine a single, brilliant doctor who has studied every field of medicine. For any patient that walks in, this doctor must mentally access all of their knowledge to make a diagnosis. While highly capable, this is inefficient.
* **An MoE Model is a Specialist Hospital:** Now, imagine a large hospital with a team of 64 specialists and a skilled triage nurse at the front desk.
    * **The Specialists (Experts):** You have a cardiologist, a neurologist, etc. Their combined knowledge is immense.
    * **The Triage Nurse (The Router):** When a patient (a **token**) arrives, the nurse quickly assesses their symptoms and sends them to the 2 most relevant specialists.
    * **Conditional Computation:** The patient gets expert care, while the other 62 specialists do no work.

This is the core of MoE:
* **Vast Knowledge (Parameters):** The model's total parameter count is the sum of all specialists.
* **Efficient Computation (FLOPs):** The work done for any token is small and fixed, as it only consults a small subset (`k`) of the total available experts (`N`).

This principle is called **conditional computation**—the computation performed is conditional on the input itself.

### 2. Architectural Anatomy: MoE Layer vs. Vanilla FFN

An MoE layer does not replace the FFN with a completely alien structure. Instead, it wraps multiple standard FFNs (the "experts") within a dynamic routing mechanism. Let's compare their mathematical structures for a single token input $x \in \mathbb{R}^{d_{model}}$.

#### Vanilla FFN Structure (Monolithic)
A standard FFN is a single, dense block with two linear transformations and a non-linearity.

$$y_{FFN}(x) = W_2 (\text{Activation}(xW_1 + b_1)) + b_2$$

* $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $b_1 \in \mathbb{R}^{d_{ff}}$
* $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$, $b_2 \in \mathbb{R}^{d_{model}}$
* **Nature:** Static and monolithic. Every token `x` is processed by the exact same weights, $W_1$ and $W_2$.

#### MoE Layer Structure (Composite)
An MoE layer is a composite function composed of three distinct parts: a router, multiple experts, and a combination rule.

**A. The Router:** A single linear layer that produces routing probabilities.

$$\text{Gates} = G(x) = \text{Softmax}(xW_g)$$

* $W_g \in \mathbb{R}^{d_{model} \times N}$
* **Nature:** Its output is a probability vector of size `N`, determining which experts are best suited for `x`.

**B. The Experts:** A set of `N` independent FFNs. Each expert $E_i$ has its own unique weights.

$$E_i(x) = W_{2,i} (\text{Activation}(xW_{1,i} + b_{1,i})) + b_{2,i}$$

* $W_{1,i} \in \mathbb{R}^{d_{model} \times d_{ff}}$, $b_{1,i} \in \mathbb{R}^{d_{ff}}$ for each expert `i`.
* $W_{2,i} \in \mathbb{R}^{d_{ff} \times d_{model}}$, $b_{2,i} \in \mathbb{R}^{d_{model}}$ for each expert `i`.
* **Nature:** Each expert is a specialist, with weights learned for specific types of patterns.

**C. The Final MoE Output:** A combination of the above parts.

$$y_{MoE}(x) = \sum_{i \in \text{TopK}(G(x), k)} G(x)_i \cdot E_i(x)$$

* **Nature:** Dynamic and sparse. The final computation for `x` is a weighted sum of the outputs from only the top `k` selected experts.

### 3. The Full Architecture and Mathematical Formulation

Now let's detail the full process, including the critical load balancing loss.

#### A. The Experts & B. The Gating Network (Router)
The structure is as described in the section above. For a batch of tokens, the router computes gate probabilities for every token independently.

#### C. Top-K Gating and Output Combination
We use sparse top-k gating to select the `k` experts with the highest gate probabilities for each token. The final output is a weighted sum of the outputs from these selected experts.

#### D. An Essential Detail: The Load Balancing Loss
* **The Problem (Expert Collapse):** If the model were trained only on its primary task (e.g., predicting the next word), the router would quickly become "lazy." It might discover that a few experts are slightly better on average and would start sending almost all tokens to them. This leaves the other experts under-trained and useless, causing the massive parameter count of the MoE layer to go to waste. This is known as "expert collapse."

* **The Solution:** To counteract this, we add an **auxiliary loss** to the model's total loss during training. This loss penalizes imbalanced expert utilization and encourages the router to spread tokens evenly.

* **Mathematical Formulation:** The loss is designed to increase when some experts receive a disproportionately large load. For a batch of $T = B \times L$ tokens and $N$ experts:
    1.  Define $f_i$: The fraction of tokens in the batch dispatched to expert `i`.
  
        $$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}(\text{expert chosen for token } t \text{ is } i)$$

        where $\mathbb{I}$ is the indicator function.
    2.  Define $P_i$: The average router probability assigned to expert `i` across all tokens in the batch.

        $$P_i = \frac{1}{T} \sum_{t=1}^{T} G(x_t)_i$$
    
    3.  The load balancing loss is then their dot product, scaled by the number of experts:
     
        $$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$
    
    Here, $\alpha$ is a small, tunable hyperparameter that controls the strength of this loss. By minimizing $L_{aux}$, the model is forced to keep both the dispatch assignments ($f_i$) and the router probabilities ($P_i$) relatively flat, ensuring all experts receive traffic and continue to learn.

#### Why this loss works
The core idea is that the loss function creates a **high penalty for "compounding certainty."** It punishes the router for being both *confident* about an expert and *over-utilizing* it at the same time. Minimizing this penalty forces the router to spread its bets.

To see why, let's look at two contrasting scenarios for a batch of tokens and a model with `N=8` experts.

---

### Scenario 1: The "Bad" Case - Expert Collapse

Imagine the router has become "lazy" and learned that Expert #3 is pretty good for most things. It develops a strong preference.

* **Router Behavior:** For almost every token, the router outputs a probability distribution that heavily favors Expert #3. For a typical token, the probabilities might look like this:
    `[0.05, 0.05, **0.70**, 0.05, 0.05, 0.05, 0.05, 0.05]`
* **Calculating `P_i` (Average Router Probability):**
    When we average these probabilities over the whole batch, the `P_i` vector will reflect this bias.
    * `P_3` will be very high (e.g., **~0.7**).
    * All other `P_i` values will be very low (e.g., **~0.04** each).
* **Calculating `f_i` (Actual Dispatch Fraction):**
    Since Expert #3 consistently gets the highest score, the `top-k` selection (let's say `k=1`) will choose it almost every time.
    * `f_3` will be very high (e.g., **~0.95**, meaning 95% of tokens went to it).
    * All other `f_i` values will be very low (e.g., close to **0**).
* **Calculating the Loss Term (`f_i * P_i`):**
    Now, let's look at the product that makes up the loss.
    * **For Expert #3:** `f_3 * P_3` is `high * high = VERY HIGH` (e.g., `0.95 * 0.7 = 0.665`).
    * **For all other experts:** `f_i * P_i` is `low * low = VERY LOW` (e.g., `0.0 * 0.04 = 0.0`).
* **Total Loss:** The total auxiliary loss, $\sum f_i \cdot P_i$, is dominated by the single, massive term from Expert #3. The loss value is **high**.

The model's optimizer sees this high loss and is forced to adjust the router's weights (`W_g`) to reduce it.

---

### Scenario 2: The "Ideal" Case - A Balanced Router

Now, imagine a well-trained, balanced router. It has learned that different experts are useful for different tokens and distributes its confidence accordingly.

* **Router Behavior:** Over a diverse batch of tokens, no single expert is consistently favored. The router's probabilities are spread out.
* **Calculating `P_i` (Average Router Probability):**
    Because the confidence is spread evenly across the batch, the average probability for each expert will be roughly the same.
    * Every `P_i` will be close to the ideal uniform value of `1/N` (e.g., **~0.125** for `N=8`).
* **Calculating `f_i` (Actual Dispatch Fraction):**
    Since no expert has a consistently high probability, the `top-k` selections will also be distributed across all experts over the course of the batch.
    * Every `f_i` will also be roughly equal (e.g., close to **~0.125** for `k=1`).
* **Calculating the Loss Term (`f_i * P_i`):**
    * **For every expert:** `f_i * P_i` is `low * low = VERY LOW` (e.g., `0.125 * 0.125 = 0.0156`).
* **Total Loss:** The total loss is the sum of many small, similar terms. For our example, `8 * 0.0156 = 0.125`. This value is **much lower** than the `0.665` from the collapsed scenario.

### The Intuitive Conclusion

The loss function $L_{aux} \propto \sum f_i \cdot P_i$ is minimized when the product $f_i \cdot P_i$ is small for all `i`.

A large value of this product can **only** occur if an expert `i` has *both* a high dispatch fraction (`f_i`) and a high average router probability (`P_i`). This is the exact mathematical signature of expert collapse.

By penalizing this specific condition, the loss function creates a gradient that pushes the router away from this state. To minimize the loss, the router must reduce its confidence (`P_i`) in any expert that is being over-utilized (`f_i` is high). This naturally encourages it to explore other experts, increasing their `P_i` and `f_i` values in turn. The stable equilibrium point that minimizes this loss is a **diversified, flat distribution** where all experts are utilized roughly equally.

Think of it as a "portfolio diversification" penalty. The loss is low if your investments are spread out. The loss is high if you put all your money (`f_i`) into a stock you are already overly confident in (`P_i`). To minimize risk (loss), you must diversify.
#### Alternative solutions for load  balancing loss

1- Maximize the entropy (minimize negative entropy)

$$
loss = \sum_i^N p_i \log p_i
$$

2- minimize variance

$$
\mathcal{L}_{\text{balance}} = \left( \frac{1}{N} \sum_{i=1}^N p_i \right)^2 - \frac{1}{N} \sum_{i=1}^N p_i^2
= \text{mean}(p)^2 - \text{mean}(p^2)
$$

### 4. Full Complexity Analysis

* **Parameter Count:** The parameters are the sum of the router and all $N$ experts.
    
  $$
  P_{MoE} = (d_{model} \cdot N) + (N \cdot P_{FFN})
  $$
    
This is enormous. For a model with 64 experts, the FFN parameters are roughly 64 times larger than the baseline.
* **Computational Complexity (FLOPs):** This is the magic of MoE. The computation is decoupled from the total number of experts, $N$.
* Router FLOPs: $B \cdot L \cdot (2 \cdot d_{model} \cdot N)$
* Expert FLOPs: $B \cdot L \cdot k \cdot (4 \cdot d_{model} \cdot d_{ff})$  (Each token is processed by $k$ experts)
* **Total FLOPs:** $\approx B \cdot L \cdot (2 \cdot d_{model} \cdot N + 4 \cdot k \cdot d_{model} \cdot d_{ff})$. The expert computation dominates. Crucially, the FLOPs scale with $k$ (e.g., 2), not $N$ (e.g., 64).

 
### 5. Code Implementation

This updated code snippet now calculates and returns the auxiliary load balancing loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Expert(nn.Module):
    # Same as before
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, k: int, aux_loss_alpha: float = 0.01):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.d_model = d_model
        self.aux_loss_alpha = aux_loss_alpha

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        num_tokens = x_flat.size(0)

        router_logits = self.gate(x_flat)
        gates_softmax = F.softmax(router_logits, dim=-1)
        
        # --- Load Balancing Loss Calculation ---
        # f_i: Fraction of tokens dispatched to expert i
        # P_i: Average router probability for expert i
        P = gates_softmax.mean(dim=0)
        
        # Calculate a temporary dispatch tensor for calculating f_i
        temp_dispatch = torch.zeros_like(gates_softmax)
        # top_k_indices has shape (num_tokens, k)
        top_k_gates, top_k_indices = torch.topk(gates_softmax, self.k, dim=-1)
        # scatter_ will place a 1 at the chosen expert indices
        temp_dispatch.scatter_(1, top_k_indices, 1) 
        f = temp_dispatch.mean(dim=0)
        
        # Loss is N * dot_product(f, P) scaled by alpha
        load_balancing_loss = self.aux_loss_alpha * self.n_experts * torch.dot(f, P)
        # --- End of Loss Calculation ---
        
        top_k_gates_norm = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.k)
        flat_expert_indices = top_k_indices.flatten()
        
        batched_expert_inputs = x_flat[flat_token_indices]
        gated_inputs = batched_expert_inputs * top_k_gates_norm.flatten().unsqueeze(-1)
        
        dispatched_outputs = torch.zeros_like(gated_inputs)
        
        for i in range(self.n_experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                expert_inputs = gated_inputs[mask]
                dispatched_outputs[mask] = self.experts[i](expert_inputs)
        
        output.scatter_add_(0, flat_token_indices.unsqueeze(-1).expand(-1, D), dispatched_outputs)

        return output.view(B, L, D), load_balancing_loss
```

### 6. Key References

* **Conceptual Foundation for Sparse MoE:**
    Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv preprint arXiv:1701.06538.
* **Modern Scaling and Implementation (including load balancing):**
    Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv preprint arXiv:2101.03961.


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
 


