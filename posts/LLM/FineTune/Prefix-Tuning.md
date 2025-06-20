[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/FineTuning-Comprehensive_Tutorial_on_Finetuning_LLMs-orange?style=for-the-badge&logo=github)](../../FineTuning)

# Prefix-Tuning

In the landscape of Parameter-Efficient Fine-Tuning (PEFT), methods like Adapter Tuning modify the model's architecture by injecting new layers. Prefix-Tuning proposes an even less invasive idea: what if we could achieve specialized behavior without touching the model's architecture at all?

The core idea is to **freeze the entire pre-trained LLM** and learn a small sequence of special, continuous vectors—a "prefix"—that we prepend to the input. This learned prefix acts as an optimized set of instructions that steers the frozen LLM's attention and directs it to perform the desired task.

#### **Example & Intuition**

Imagine the powerful, pre-trained LLM is a genie, capable of incredible feats but needing very precise instructions. Prefix-Tuning is like learning the perfect "magic words" to say to the genie. You don't change the genie (the LLM); you just learn the perfect phrase to prepend to any request to get the exact behavior we want.

#### **Use Case Scenario**

The goal is to efficiently specialize a single base LLM for many tasks, especially generative ones where control over the output style, format, or topic is crucial.

* **Content Generation Factory:** A marketing firm uses a single, powerful base model (like Llama 3). They have several learned prefixes:
    * When they need a professional and formal blog post, they prepend the **"formal-blog-prefix"** to the topic.
    * When they need a witty and engaging tweet, they prepend the **"twitter-wit-prefix"** to the same topic.
* The same base model produces vastly different outputs based on the small (a few kilobytes) prefix it's given, saving immense storage and computational resources.

---
### How It Works: A Detailed Breakdown

#### 1. The Architecture: Prepending a "Virtual" Prompt

The key architectural principle is that the pre-trained LLM is **100% frozen**. The only new, trainable components are the prefix parameters.

##### The Journey from `P_θ` to `P_k` and `P_v`

1.  **One `P_θ` per Layer, Used for Both K and V:** For each layer in the Transformer, we define **one** learnable prefix matrix, `P_θ`. This is the raw source material. Its shape is `[prefix_length, d_model]`. Because lower layers handle syntax and higher layers handle semantics, each layer gets its own unique, learnable `P_θ` to provide guidance at the correct level of abstraction. The total set of trainable parameters is a stack of these matrices.

2.  **Projection, Not Concatenation:** The raw `P_θ` matrix is **not** concatenated with the weight matrices `W_K` and `W_V`. Instead, `P_θ` is **projected by** `W_K` and `W_V` using matrix multiplication. The LLM's existing (and frozen) projection matrices are reused for this.

    The process for a single layer is:
    * **Generate Key Prefix:** `P_k_raw = P_θ @ W_k`
    * **Generate Value Prefix:** `P_v_raw = P_θ @ W_v`

3.  **Final Concatenation with Text's K and V:** *After* the prefixes `P_k` and `P_v` have been created, *they* are then concatenated with the `K_text` and `V_text` generated from the actual user input.

##### A Look at the Tensor Dimensions

Let's use concrete numbers for a model like Llama 3 8B:
* `d_model`: 4096
* `num_heads`: 32
* `d_head`: 128
* `prefix_length`: 10
* `sequence_length`: 50 (length of user's text)

The final key matrix passed to the attention calculation, `K_final = concat(P_k, K_text)`, will have the shape `[batch_size, num_heads, prefix_length + sequence_length, d_head]` -> `[4, 32, 60, 128]`.

#### 2. The Mathematics

The modified attention calculation at layer `i` is:
$$h_i = \text{Attention}(Q_i, [P_{k,i}; K_i], [P_{v,i}; V_i])$$
where `[;]` denotes concatenation and $P_{k,i}$ and $P_{v,i}$ are derived from that layer's specific $P_{\theta, i}$ being projected through the frozen $W_{k,i}$ and $W_{v,i}$.

#### 3. The Training Process

##### Prefix Initialization: A Critical Step for Stability
You cannot initialize the prefix with large, random values, as this would introduce "noise" and disrupt the carefully calibrated weights of the frozen LLM, destabilizing training from step one. The initialization strategy is key to providing a good starting point for the optimization process.

* **Initialization Method 1: Using Real Word Embeddings (The "Warm Start")**
    * **Concept:** The most effective method suggested by the original paper is to initialize the prefix parameters using the pre-trained embeddings of actual vocabulary words that are relevant to the task.
    * **Example:** For a summarization task and a prefix of length 4, you might choose the tokens `"Summarize", "this", "text", ":"`. You would look up the embedding vectors for these four words in the LLM's frozen embedding table and use them as the initial values for your `P_θ` matrix (`shape [4, d_model]`).
    * **Benefit:** This gives the model a sensible, stable starting point that is already in the correct "semantic region" for the task, which can lead to faster convergence.

* **Initialization Method 2: Small Random Values**
    * **Concept:** A simpler, common alternative is to initialize the `P_θ` matrix from a normal distribution with a mean of 0 and a small standard deviation (e.g., 0.02).
    * **Benefit:** This is easy to implement and still ensures that the initial prefix doesn't create large, disruptive activations. The model then learns the appropriate values from scratch during training.

* **Do we initialize all the layers the same?**
    **No, not typically.** While you *could* use the same initialization values for the `P_θ` matrix at each layer, it is generally not done, and for good reason. Each `P_θ_i` (for layer `i`) is an independent set of trainable parameters.

    1.  **If using Real Word Embeddings:** You would typically use the same set of word embeddings (e.g., from "Summarize this text:") to initialize the `P_θ` matrix at **every layer**. Even though the starting values are the same, because each layer's `W_k` and `W_v` are different, the resulting `P_k` and `P_v` will be different. Furthermore, as training begins, the gradients flowing back to each `P_θ_i` will be unique, so they will immediately diverge and learn different, layer-specific functions.
    2.  **If using Random Values:** Each layer's `P_θ_i` matrix would be initialized independently from the random distribution. They start with different small random values and learn their own specialized roles from there.

The key takeaway is that while the *initialization strategy* is the same for all layers, the resulting prefixes are **trained independently** and evolve to serve the unique needs of their specific layer in the Transformer hierarchy.

##### The Training Loop
The process involves a standard forward pass, but with the prefix injection logic inside each attention block. The **Masked Cross-Entropy Loss** is calculated on the response tokens, and crucially, backpropagation **only updates the `P_θ` matrices** for each layer.

#### 4. Inference
At inference time, the frozen LLM is loaded along with the small, task-specific prefix matrices. When a user provides a prompt, the learned prefixes are injected into the attention calculations at each layer, steering the model to generate the desired output.

---

### Conceptual Code: From-Scratch Attention with Prefix-Tuning

This Python code (using PyTorch) shows the core logic inside a Multi-Head Attention module, clarifying the projection and concatenation steps.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionWithPrefix(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # These layers are part of the pre-trained, FROZEN LLM
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, p_theta_layer=None, attention_mask=None):
        """
        x: input text embeddings, shape [batch_size, seq_len, d_model]
        p_theta_layer: The raw prefix parameters for this layer, shape [prefix_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project input text embeddings into Q, K, V
        Q_text = self.w_q(x)
        K_text = self.w_k(x)
        V_text = self.w_v(x)

        # Reshape for Multi-Head Attention
        Q_text = Q_text.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K_text = K_text.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V_text = V_text.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # === CORE PREFIX-TUNING LOGIC ===
        if p_theta_layer is not None:
            prefix_len = p_theta_layer.shape[0]
            
            # 1. Project P_theta using the frozen W_k and W_v matrices
            with torch.no_grad(): # Ensure W_k/W_v are not updated
                P_k_raw = self.w_k(p_theta_layer)
                P_v_raw = self.w_v(p_theta_layer)

            # 2. Reshape prefixes for multi-head attention and expand for batch
            P_k = P_k_raw.view(1, prefix_len, self.num_heads, self.d_head).transpose(1, 2).expand(batch_size, -1, -1, -1)
            P_v = P_v_raw.view(1, prefix_len, self.num_heads, self.d_head).transpose(1, 2).expand(batch_size, -1, -1, -1)
            
            # 3. Concatenate prefixes with the text's K and V
            K = torch.cat([P_k, K_text], dim=2)
            V = torch.cat([P_v, V_text], dim=2)
        else:
            K = K_text
            V = V_text
        
        # Standard attention calculation with the modified K and V
        attention_scores = (Q_text @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if attention_mask is not None:
            if p_theta_layer is not None:
                 prefix_len = p_theta_layer.shape[0]
                 attention_mask = F.pad(attention_mask, (prefix_len, 0), value=False)
            attention_scores = attention_scores.masked_fill(attention_mask == 1, -1e9)
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = attention_probs @ V
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
```

---
### References

* **Original Prefix-Tuning Paper:** Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics.*
* **Prompt Tuning:** Lester, B., Al-Rfou, R., & Constant, N. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.* A related, slightly simpler method where a prefix is only learned for the input embedding layer, rather than for every layer.
