[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/FineTuning-Comprehensive_Tutorial_on_Finetuning_LLMs-orange?style=for-the-badge&logo=github)](../../FineTuning)

# Prompt Tuning

Prefix-Tuning was a powerful idea: steer a frozen LLM by learning continuous "virtual tokens" that are prepended to the keys and values in every attention layer. However, it had some challenges. The training could sometimes be unstable, and its performance wasn't always as strong as full fine-tuning on harder, smaller-scale datasets.

**Prompt Tuning** was developed to address these issues. It adopts the core concept of using continuous prompts at every layer but implements it in a more robust and "deeper" way.

#### **Example & Intuition**

Let's revisit our "genie" analogy for the frozen LLM.

* **Prefix-Tuning:** This was like learning a set of "magic words" to whisper *alongside* your request. The genie hears both your request and the magic words simultaneously and is steered by them.
* **Prompt Tuning:** This is more like having a set of magical "tuning forks." You don't just whisper magic words at the start; at *every step* of the genie's thought process, you strike a specific tuning fork. This continuously resonates through the genie's mind, keeping its thoughts perfectly aligned with your desired task from beginning to end. It's a deeper, more pervasive form of guidance.

The key difference is that Prompt Tuning injects the influence of the learned prompts not just at the attention level, but throughout the entire sequence of processed tokens at each layer.

#### **Use Case Scenario**

The goal is the same as other PEFT methods—efficiently adapting a single base model to many tasks—but Prompt Tuning aims to be a more universal solution that works well across different model sizes (from 300M to 10B+ parameters) and across a wider range of challenging tasks, including sequence tagging like Named Entity Recognition (NER).

* **A Unified NLP Platform:** A company in San Jose wants to use a single Llama 3 8B model for multiple purposes.
    * For their customer support chatbot, they load the **"SFT_prompts"**.
    * For extracting company names and dates from legal documents (`NER`), they load the **"NER_prompts"**.
    * For classifying incoming emails as `Spam` or `Not Spam`, they load the **"Classifier_prompts"**.
* Prompt Tuning provides a robust method that performs strongly on both generative (chat) and classification/tagging (NER) tasks, making it a highly versatile choice.

---
### How It Works: A Detailed Breakdown

#### 1. The Architecture: "Deep" Prompt Tuning

Like Prefix-Tuning, the pre-trained LLM is **100% frozen**. The innovation lies in *how* and *where* the learnable prompts are used.

* **The Core Idea:** Instead of prepending prompts only to the key and value matrices inside the attention block, Prompt Tuning treats the learnable prompts as if they were actual input tokens. It prepends these "virtual tokens" to the sequence of word embeddings at the very beginning, and then **allows them to be processed through all subsequent layers of the Transformer**.

* **The Flow:**
    1.  **Input Layer:** A sequence of learnable prompt embeddings `P_θ` (e.g., `prefix_length=20`) is concatenated with the actual text embeddings `x` (e.g., `seq_len=100`). The input to the *first* Transformer layer is this combined sequence `[P_θ, x]`.
    2.  **Subsequent Layers:** The output from Layer 1, which now has a length of `prefix_length + seq_len`, becomes the *full input* to Layer 2. The entire combined sequence is processed by the attention and FFN blocks. This continues for all layers.
    3.  **No More Key/Value Injection:** Unlike Prefix-Tuning, there is no special logic *inside* the attention block. The attention mechanism simply operates on the combined sequence it receives. The prompt embeddings influence the text embeddings naturally through the standard self-attention process.

This "deep" approach, where the prompt embeddings evolve layer by layer, proved to be more stable and effective than the "shallow" approach of injecting a fixed prefix at each layer's attention block.

#### 2. The Mathematics

Let the input text embeddings be $X_{emb} \in \mathbb{R}^{L \times d}$ (where `L` is sequence length, `d` is `d_model`). Let the trainable prompt embeddings be $P_{emb} \in \mathbb{R}^{P \times d}$ (where `P` is prefix length).

The input to the first Transformer layer ($h_0$) is the concatenation:
$$h_0 = [P_{emb}; X_{emb}]$$

The output of any subsequent layer `i` is then calculated standardly:
$$h_i = \text{TransformerLayer}_i(h_{i-1})$$

The crucial difference is that the learnable parameters `P_emb` are part of the input sequence from the very beginning, and their representations are updated and transformed at each layer, just like real word embeddings.

#### 3. The Training Process

* **Initialization:** The prompt embeddings `P_θ` are initialized. This is less sensitive than in Prefix-Tuning, but initialization from a random normal distribution with a small variance is standard.
* **Input-Output Pairs:** The data is identical to what's used for full fine-tuning (e.g., `(instruction, response)` for SFT).
* **The Forward Pass:** The prompt embeddings and text embeddings are concatenated and passed through the entire Transformer stack.
* **Loss Function:** The loss is calculated only on the outputs corresponding to the **text portion** of the sequence. For a generative task, you would use **Masked Cross-Entropy Loss**, ignoring the outputs generated from the positions of the virtual prompt tokens.
* **Backpropagation:** The gradients from the loss (calculated only on the text outputs) flow back through the entire frozen model. Because the prompt embeddings were part of the computation at every layer, they receive gradients and are updated. The frozen LLM parameters do not have `requires_grad=True` and are not passed to the optimizer, so they remain unchanged.

#### 4. Inference
1.  Load the frozen, pre-trained base LLM.
2.  Load the small, task-specific prompt parameter matrix `P_θ`.
3.  For a new user prompt, convert it to embeddings, prepend the learned prompt embeddings `P_θ`, and run a single forward pass through the entire model.
4.  The model generates the output autoregressively, with its behavior steered by the influence of the virtual prompt tokens from the very first layer.

---

### Conceptual Code: From-Scratch Prompt Tuning

This implementation shows how simple the architecture is. There is no need for a custom attention module; we just need to concatenate the prompts at the beginning.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerLayer(nn.Module):
    """A standard Transformer layer without any special modifications."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # In a real model, this would use MultiHeadAttention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Standard self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Standard FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class PTuningV2Model(nn.Module):
    """A model that implements Prompt Tuning."""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, prefix_len):
        super().__init__()
        self.prefix_len = prefix_len
        
        # --- Base Model Components (Frozen) ---
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

        # --- Trainable Prompt Parameters ---
        self.prompt_embeddings = nn.Parameter(torch.randn(prefix_len, d_model))

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        
        # 1. Get text embeddings from the input IDs
        text_embeds = self.word_embeddings(input_ids)
        
        # 2. Expand the learned prompts to match the batch size
        # Shape: [batch_size, prefix_len, d_model]
        expanded_prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. CORE OF Prompt Tuning: Concatenate prompts and text embeddings at the start
        # Shape: [batch_size, prefix_len + seq_len, d_model]
        x = torch.cat([expanded_prompts, text_embeds], dim=1)
        
        # 4. Pass the combined sequence through all Transformer layers
        for layer in self.layers:
            x = layer(x) # Using a standard transformer layer
            
        # 5. Get final logits from the output layer
        logits = self.output_layer(x)
        
        # 6. IMPORTANT: Slice the logits to return only predictions for the text part
        # We don't care about the model's output for the prompt positions
        text_logits = logits[:, self.prefix_len:, :]
        
        return text_logits

# --- Conceptual Training Loop ---
if __name__ == '__main__':
    # Hyperparameters
    vocab_size, d_model, num_layers, num_heads, d_ff = 50257, 768, 12, 12, 3072
    prefix_len, seq_len, batch_size = 20, 100, 4

    # Instantiate the model
    model = PTuningV2Model(vocab_size, d_model, num_layers, num_heads, d_ff, prefix_len)

    # Freeze all parameters except the prompt embeddings
    for name, param in model.named_parameters():
        if 'prompt_embeddings' not in name:
            param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    print("Starting conceptual training loop for Prompt Tuning...")
    for epoch in range(2):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        
        # The forward pass handles the prompt concatenation internally
        text_logits = model(input_ids)
        
        # Loss is calculated only on the text logits
        loss = loss_function(text_logits.reshape(-1, vocab_size), labels.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Verify which parameters have gradients
    print("\nVerifying gradients after training step:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - Gradients exist for: {name}")

```
---

# P-Tuning v2 Tutorial: Intuition, Mathematics & Implementation**  
*An efficient parameter-efficient tuning (PEFT) method for large language models.*

---

#### **1. Core Intuition**  
- **Problem**: Full fine-tuning LLMs is expensive. Prompt tuning (trainable soft prompts) struggles with small models (<10B parameters).  
- **Solution**: P-Tuning v2 injects **layer-specific trainable prompts** at **every transformer layer**, enabling deep task adaptation with minimal parameters (0.1% of model size).  
- **Key Insight**:  
  - *Shallow tuning* (Prompt Tuning): Only input layer affected → weak signal propagation.  
  - *Deep tuning* (P-Tuning v2): Every layer adapted → complex task alignment.  
---

#### **2. Mathematical Formulation**  
Let a transformer have $L$ layers. For layer $ℓ$:  

- **Input**: Hidden states $H^{ℓ-1} ∈ ℝ^{batch \times seq_len \times d_model}$  
- **Trainable Prompts**: $P_ℓ \in ℝ^{\text(prompt_len) \times d_model}$  
- **Modified Input**:  

  $$
  \widetilde{H}^{ℓ-1} = \text{Concat}\left(P_ℓ,\  H^{ℓ-1}[:, \text{prompt\_len}:, :]\right) \in ℝ^{batch × (prompt\_len + seq\_len) × d_model}
  $$  

- **Layer Output**:  

  $$
  H^ℓ = \text{TransformerLayer}^ℓ\left(\widetilde{H}^{ℓ-1}\right)
  $$  

- **Only Parameters**: $P_1, P_2, ..., P_L$  
- **Frozen**: Original transformer weights `θ_model`.

---

#### **3. Training Process**  
**Input-Output Pairs**:  
- For classification: `(text, label)` → `("Review: This movie was great", "positive")`  
- For generation: `(input, target)` → `("Translate to French: Hello", "Bonjour")`  

**Loss Function**:  
- Cross-Entropy over task labels:  

  $$
  \mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)
  $$  
  where $\hat{y}$ = model output logits.

**Optimization**:  
- Optimizer: Adam (LR = 1e-4)  
- Train only prompt parameters $P_ℓ$.  

---

#### **4. Model Compatibility**  
| **Architecture** | **Usage**                          | **Example Models**       |
|------------------|------------------------------------|--------------------------|
| **Encoder-Decoder** | Prepend prompts to encoder/decoder inputs | T5, BART                 |
| **Decoder-Only**    | Prepend prompts to decoder inputs        | GPT-3, Llama, OPT        |
| **Encoder-Only**    | Prepend prompts to encoder inputs        | BERT, RoBERTa            |

**Best Results**:  
- Matches full fine-tuning on NLU tasks (GLUE, SuperGLUE)  
- Effective for models ≥100M parameters (unlike prompt tuning).

---

#### **5. Minimal PyTorch Implementation**  
*No high-level transformer libraries - core components only:*

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """Simplified Transformer Layer (Self-Attention + FFN)"""
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        return self.norm2(x)

class PTuningv2(nn.Module):
    """P-Tuning v2 Wrapper for a Transformer"""
    def __init__(self, transformer, num_layers, prompt_len, d_model):
        super().__init__()
        self.transformer = transformer
        # Layer-specific trainable prompts
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_len, d_model)) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: input embeddings [batch, seq_len, d_model]
        h = x
        for i, layer in enumerate(self.transformer.layers):
            # Concat layer-specific prompt
            p = self.prompts[i].expand(x.size(0), -1, -1)  # [batch, prompt_len, d_model]
            h_modified = torch.cat([p, h], dim=1)
            # Pass through transformer layer
            h = layer(h_modified)
            # Remove prompt for next layer (optional, see paper)
            h = h[:, p.size(1):, :]
        return h

# Example Usage
d_model = 512
prompt_len = 10
num_layers = 6

# Create a minimal transformer
transformer = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
model = PTuningv2(transformer, num_layers, prompt_len, d_model)

# Freeze transformer (only train prompts)
for param in transformer.parameters():
    param.requires_grad = False

# Training setup
optimizer = torch.optim.Adam(model.prompts.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop (example for classification)
for batch in dataloader:
    inputs, labels = batch
    optimizer.zero_grad()
    outputs = model(inputs)[:, 0, :]  # Use [CLS] token
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

#### **6. Key Implementation Notes**  
1. **Prompt Initialization**: Gaussian noise (`randn`) works; task-specific initialization may improve performance.  
2. **Prompt Placement**:  
   - *Encoder*: Prepend to input sequence.  
   - *Decoder*: Prepend to target sequence (autoregressive attention respects left-context).  
3. **Sequence Length**: Total length = `original_seq_len + prompt_len` (fixed).  
4. **Efficiency**: 100-1000× fewer parameters than full fine-tuning.

---

#### **7. When to Use P-Tuning v2**  
- ✅ Limited GPU memory  
- ✅ Quick task adaptation  
- ✅ Preserving pretrained knowledge  
- ❌ Extremely small datasets (under 100 examples)  

---

#### **Reference**  
Liu, X., Ji, K., Fu, Y., Du, Z., Yang, Z., & Tang, J. (2022).  
[**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**](https://arxiv.org/abs/2110.07602)  
*arXiv preprint arXiv:2110.07602*.  

**Official Code**: [THUDM/P-tuning-v2](https://github.com/THUDM/P-tuning-v2)  

This tutorial covers the essence of P-Tuning v2—combining mathematical rigor with practical implementation for real-world efficiency.

### References

* **P-Tuning-V2 Tuning Paper:** Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., & Tang, J. (2021). "P-Tuning-V2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks." *arXiv preprint arXiv:2110.07602.* This paper introduces the "deep" prompt tuning approach and demonstrates its effectiveness and stability.
