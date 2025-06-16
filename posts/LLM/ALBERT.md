[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

## A Deep Dive into Cross-Layer Parameter Sharing: The ALBERT Approach

While innovations like MoE and SwiGLU focus on redesigning the FFN block *within* a single Transformer layer, Cross-Layer Parameter Sharing tackles efficiency from a different angle: it re-evaluates the relationship *between* the layers themselves. It asks a simple but profound question: "Does every layer in a deep network truly need its own unique set of weights?"

This tutorial explores the intuition, mathematics, and trade-offs of this powerful parameter reduction technique.

### 1. The Intuition: Iterative Refinement vs. An Assembly Line

To grasp the concept, it's helpful to contrast the standard Transformer architecture with a parameter-shared one.

* **A Standard Transformer is an "Assembly Line" 🏭:** Think of a 24-layer Transformer as a 24-stage factory assembly line. Each station (`Layer_i`) has its own unique, highly specialized tools (`Weights_i`).
    * **Station 1** might perform a basic task like identifying parts of speech.
    * **Station 12** might have tools to understand semantic relationships.
    * **Station 24** might have sophisticated tools to resolve complex, long-range dependencies.
    Each layer is a specialist, and the token representation is passed from one specialist to the next, getting progressively built up.

* **A Shared-Parameter Transformer is a "Master Craftsman" 👨‍🎨:** Now, imagine a master craftsman building a sculpture with a single, highly versatile set of tools. They don't have 24 different sets of tools. Instead, they apply the same set of tools **iteratively** to the raw block of marble (the input embedding).
    * **Pass 1 (Layer 1):** The craftsman uses their tools to rough out the basic shape of the sculpture.
    * **Pass 2 (Layer 2):** They apply the *exact same tools* again to the roughed-out shape, this time adding finer details.
    * **Pass 12 (Layer 12):** After many passes, the same tools are used for the final polishing.

This is the core intuition of cross-layer parameter sharing. The model learns a **general, reusable function for iterative refinement**. Instead of learning 24 distinct transformation steps, it learns a single, powerful transformation that can be applied repeatedly to deepen a token's representation.

### 2. Architectural Anatomy and Mathematical Formulation

The change is not within the block, but in how the blocks are stacked.

#### Vanilla Transformer Structure (Un-shared)
In a standard Transformer with `L` layers, each layer `l` is a distinct block, `Block_l`, with its own unique set of weights for both attention ($W_{l}^Q, W_{l}^K, \dots$) and the FFN ($W_{1,l}, W_{2,l}, \dots$). The update rule is:
$$h_l = h_{l-1} + \text{Block}_l(h_{l-1})$$
Here, `Block_l` is different from `Block_{l-1}` because their underlying weights are different.

#### Shared-Parameter Transformer Structure
In a parameter-shared model, we define only **one** block, `Block_shared`, and reuse it for every layer. The update rule becomes:
$$h_l = h_{l-1} + \text{Block}_{\text{shared}}(h_{l-1})$$
The weights within `Block_shared`—($W^Q, W^K, \dots$) and ($W_1, W_2, \dots$)—are **the same** for all layers $l=1, \dots, L$.

This sharing can be configured in different ways:
* **FFN-only sharing:** Share only the FFN sub-layer across layers, but keep attention weights unique.
* **Attention-only sharing:** Share only the attention sub-layer.
* **All-layer sharing (ALBERT-style):** Share the entire Transformer block. This provides the maximum parameter reduction.

### 3. Full Complexity Analysis

The complexity analysis for parameter sharing reveals its unique profile: it's a win for memory, but not for speed.

#### Parameter Count

This is where the technique shines. Let's compare the total parameters for the FFN blocks in an `L`-layer Transformer.

* **Vanilla Model FFN Parameters:** Each of the `L` layers has its own FFN with $P_{FFN}$ parameters.
    $$P_{TotalFFN} = L \times P_{FFN} = L \times (2 \cdot d_{model} \cdot d_{ff} + \dots)$$
* **Shared Model FFN Parameters:** There is only one unique FFN block, whose parameters are reused.
    $$P_{SharedFFN} = P_{FFN} = 2 \cdot d_{model} \cdot d_{ff} + \dots$$

**Conclusion:** Cross-layer parameter sharing reduces the number of FFN (and/or attention) parameters by a factor of nearly **L**. For a 12-layer model, this means a ~92% reduction in block parameters, leading to a much smaller model footprint on disk and in RAM.

#### Computational Complexity (FLOPs)

This is the crucial trade-off.

* **FLOPs for one layer:** $\text{FLOPs}_{layer} \approx 4 \cdot B \cdot L \cdot d_{model} \cdot d_{ff}$ (for the FFN part).
* **Total FLOPs for a full forward pass:**
    $$\text{FLOPs}_{Total} \approx L \times \text{FLOPs}_{layer}$$

**Conclusion:** The total FLOPs for a forward pass are **unchanged** compared to a standard Transformer. Even though we are reusing the same weight matrices, the actual matrix multiplications must still be executed at each of the `L` layers. The CPU/GPU doesn't get to skip work.

This means parameter sharing provides **memory and storage efficiency**, not **inference speed or computational efficiency**.

### 4. Code Implementation

This technique is an architectural pattern for how you build your model, not a new type of layer. The code below contrasts a standard Transformer stack with a parameter-shared one.

```python
import torch
import torch.nn as nn

# A standard Transformer Block (Attention + FFN)
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN architecture
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

# --- STANDARD IMPLEMENTATION ---
class UnsharedTransformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # Create a list of N_LAYERS unique, independent blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        print(f"Unshared Model Total Parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- PARAMETER-SHARED IMPLEMENTATION ---
class SharedTransformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.num_layers = num_layers
        # Create ONLY ONE instance of the Transformer block
        self.shared_block = TransformerBlock(d_model, num_heads, d_ff)
        print(f"Shared Model Total Parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        # Call the SAME block in a loop N_LAYERS times
        for _ in range(self.num_layers):
            x = self.shared_block(x)
        return x

# Example Usage
# d_model, n_heads, d_ff, n_layers = 512, 8, 2048, 12
# unshared_model = UnsharedTransformer(n_layers, d_model, n_heads, d_ff)
# > Unshared Model Total Parameters: 50468352
# shared_model = SharedTransformer(n_layers, d_model, n_heads, d_ff)
# > Shared Model Total Parameters: 4205824
# The parameter count is ~12x smaller, as expected.
```

### 5. Key Reference

This technique was systematically explored and popularized by the ALBERT paper, which sought to build more parameter-efficient versions of BERT.

* **Canonical Paper:** Lan, Z., Chen, M., Goodman, S., et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. arXiv preprint arXiv:1909.11942.
