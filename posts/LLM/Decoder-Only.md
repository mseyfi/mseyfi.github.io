[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

## A Deep Dive into Decoder-Only Transformers

*Last Updated: June 8, 2025*

### 1. The Task and Core Concept: Autoregressive Generation

The core task of a decoder-only model is **autoregressive text generation**. The model learns to predict the very next word given a sequence of preceding words. It generates text one token at a time, feeding its own previous output back in as input to predict the next token. This simple, self-supervised objective, when applied at a massive scale, enables the model to learn grammar, facts, reasoning abilities, and style.

---
### 2. Architecture: The Power of Masking

A decoder-only model is a stack of decoder layers from the original Transformer architecture. Its critical component is **Causal Self-Attention** (or Masked Self-Attention), which ensures that the prediction for a token at position `i` can only depend on the known outputs at positions less than `i`.

This is achieved by adding a mask to the attention score matrix before the softmax operation, effectively zeroing out the probabilities for all future tokens.

#### The Mathematics of Causal Self-Attention

We modify the standard attention formula by adding a mask matrix $M$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

#### Code Example: Generating the Causal Attention Mask
Here is a practical PyTorch snippet demonstrating how this mask is generated on the fly.

```python
import torch

# Define the sequence length for a given batch
sequence_length = 4

# Create the mask that prevents attending to future tokens
# `torch.triu` creates an upper triangular matrix. diagonal=1 ensures the diagonal is 0.
causal_mask_base = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])

# The final mask uses a large negative number for masked positions and 0.0 for unmasked.
final_mask = torch.zeros(sequence_length, sequence_length)
final_mask.masked_fill_(causal_mask_base, float('-inf'))
# tensor([[0., -inf, -inf, -inf],
#         [0.,   0., -inf, -inf],
#         [0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.]])

# This final_mask is then added to the attention scores before the softmax.
```

---
### 3. Positional Embeddings: Encoding Order
The Transformer architecture is permutation-invariant. We must explicitly inject positional information. The final input embedding for each token is the sum of its token embedding and its positional embedding.

#### 3.1 Absolute Positional Embeddings (e.g., GPT-3)
* **Mechanism:** A unique vector is learned for each absolute position in the sequence (1st, 2nd, 3rd, etc.) from an embedding matrix of size `[max_sequence_length, hidden_size]`.
* **Limitation:** This method has a hard maximum sequence length and may not generalize well beyond it.

#### 3.2 Deep Dive: Rotary Positional Embeddings (RoPE)
Modern state-of-the-art models like Llama use a more advanced method called Rotary Positional Embeddings (RoPE).

* **Intuition:** Instead of *adding* position information, RoPE *rotates* our Query and Key vectors based on their position. The key insight is that the dot product between two vectors is related to the cosine of the angle between them. If we rotate our Query vector by an angle `m` and our Key vector by an angle `n`, the dot product between the new vectors will depend on the *relative angle* `m-n`. By making the rotation angle a function of a token's position, the attention score naturally becomes dependent on the relative positions of the tokens, not their absolute ones.

* **Mathematical Formulation:**
    1.  First, we view our `d`-dimensional embedding vectors (Queries and Keys) as a sequence of `d/2` two-dimensional vectors. Let's take one such pair $(x_{2i}, x_{2i+1})$ from a vector $x$ at position $m$.
    2.  We define a rotation angle $\theta_{m,i} = m \cdot 10000^{-2i/d}$. The angle depends on both the token's position `m` and the feature index `i`. Different dimensions are rotated at different frequencies.
    3.  We apply a 2D rotation matrix to this pair:
        $$R_m = \begin{pmatrix} \cos(\theta_{m,i}) & -\sin(\theta_{m,i}) \\ \sin(\theta_{m,i}) & \cos(\theta_{m,i}) \end{pmatrix}$$
        The positionally-encoded feature pair is then computed as $R_m \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$.
    4.  This is done for both the Query vector $q_m$ at position `m` and the Key vector $k_n$ at position `n`. The crucial property is that the dot product of the rotated vectors, $\langle R_m q_m, R_n k_n \rangle$, can be shown through trigonometric identities to be a function of their original dot product and their relative position, $m-n$.

* **Benefits:** This elegant method provides robust relative position information and has shown excellent performance when generalizing to sequence lengths far beyond what the model saw during training.

---
### 4. Training: The Next-Token Prediction Objective

#### 4.1 Tokenization and Input-Output Pairs
Using a sub-word tokenizer like BPE, the training data is created by simply shifting the tokenized text sequence by one position.

* **Input (`X`):** `[The, quick, brown, fox, jumps]`
* **Label (`Y`):** `[quick, brown, fox, jumps, .]`

#### 4.2 Loss Function: Cross-Entropy
The model is trained to minimize the **Cross-Entropy Loss** between its predicted probability distribution for the next token and the one-hot encoded vector of the true next token. For a single sequence $X = (x_1, ..., x_S)$, the loss is the average negative log-likelihood of the true next token at each position:

$$L_{CE}(\theta) = - \frac{1}{S} \sum_{t=1}^{S} \log P(x_t | x_{<t}; \theta)$$

#### 4.3 Evaluation Metric: Perplexity (and its relation to Loss)
While we use Cross-Entropy as the *loss function to be minimized during training*, we often use **Perplexity (PPL)** as the primary *metric for evaluating and comparing* language models.

* **Relationship:** Perplexity is the direct exponential of the cross-entropy loss.

    $$\text{Perplexity} = e^{L_{CE}} = \exp\left(- \frac{1}{S} \sum_{t=1}^{S} \log P(x_t | x_{<t}; \theta)\right)$$

* **Intuition: Why "Perplexity"?** It is a measure of how "perplexed" or "surprised" the model is by the test data. A lower perplexity indicates that the model is less surprised, meaning it consistently assigns a higher probability to the correct next token. A perplexity of **100** can be interpreted as the model being, on average, as uncertain about the next word as if it were choosing uniformly from 100 different words. A perfect model would have a perplexity of 1.
* **In Summary:** We train the model by minimizing cross-entropy loss. We evaluate and compare models by reporting their perplexity. **Lower is better for both.**

---
### 5. Prominent Architectures: GPT and Llama Families
#### The GPT (Generative Pre-trained Transformer) Family
Developed by OpenAI, this family pioneered the scaling of decoder-only models, trained on datasets like BookCorpus and WebText, culminating in GPT-3's demonstration of **in-context learning**.

#### The Llama (Large Language Model Meta AI) Family
Developed by Meta AI, the Llama family focused on training efficiency, demonstrating that smaller models trained on vast amounts of data (1-2 Trillion tokens from sources like C4) could outperform larger models. Their open release spurred massive innovation, and Llama 2 incorporated RLHF for safety.

#### Key Differences: Llama vs. GPT
| Feature | GPT Family (e.g., GPT-3) | Llama Family (e.g., Llama 2) |
| :--- | :--- | :--- |
| **Access** | Closed-source, accessible via API. | Open-source (weights available), allowing local deployment and research. |
| **Key Innovation Focus** | Scaling parameters to achieve emergent abilities (in-context learning). | Training efficiency (more tokens for smaller models), open access, and RLHF for safety. |
| **Architectural Tweaks** | Standard Transformer decoder. | **Pre-normalization (RMSNorm):** Normalizes inputs to layers for better training stability. |
| **Activation Function** | ReLU | **SwiGLU:** A variant of the Gated Linear Unit, which has shown better performance than ReLU. |
| **Positional Embeddings** | Learned, absolute positional embeddings. | **Rotary Positional Embeddings (RoPE):** Encodes relative position, improving long-sequence performance. |

---
### 6. Fine-Tuning and Inference
* **Fine-Tuning:** A "base model" is further trained on a curated dataset of `(instruction, desired_output)` pairs to align its behavior.
* **Inference:** Generating text is an iterative process. The strategy we use to *select* a token from the model's probability distribution is critical.

#### 6.1 A Deep Dive into Sampling Strategies

Of course, professor. That is an excellent point. A high-level description is insufficient for a graduate-level understanding. The precise mechanics of *how* a token is selected from a probability distribution are fundamental to the behavior of these models.

Let's dedicate a detailed section to this, complete with procedural explanations and code snippets. I will add this to our definitive tutorial.

---

### A Deep Dive into Inference: How the Next Token is Picked

Once the decoder model has completed its forward pass, it outputs a vector of **logits** for the next token. This vector has a size equal to the entire vocabulary (e.g., `[1, 50257]` for GPT-2). To be useful, these raw scores must be converted into a probability distribution.

**The Starting Point: The Softmax Function**

First, we apply the `softmax` function to the logits to get a probability distribution `P` where every value is between 0 and 1, and all values sum to 1.

$$P = \text{softmax}(\text{logits})$$

At this point, we have a probability for every single word in the vocabulary being the next word. The question is: **how do we choose one?**

Let's explore the common strategies, from simplest to most sophisticated.

#### 1. Greedy Search (Deterministic)

This is the most straightforward method.

* **Mechanism:** Simply select the token with the highest probability. This is equivalent to performing an `argmax` on the logits vector.
* **Intuition:** "Always choose what the model thinks is best."
* **Shortcoming:** While safe, it's extremely boring and repetitive. The model will often get stuck in loops of commonly associated words, lacking any creativity or variation. It is almost never used for creative text generation.

#### 2. Temperature Sampling

This is not a selection strategy on its own, but a modifier that affects all other sampling methods. It allows us to control the "randomness" of the model's predictions.

* **Mechanism:** We rescale the logits before applying the softmax function using a `temperature` parameter, $T$.

    $$P = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$
* **Intuition:**
    * **$T > 1$ (e.g., 1.2):** Flattens the distribution. This makes less likely tokens more probable, increasing randomness. The model becomes more "creative" and "daring."
    * **$T < 1$ (e.g., 0.7):** Sharpens the distribution. This makes high-probability tokens even more likely, reducing randomness. The model becomes more "focused" and "confident."

#### 3. Top-k Sampling

This is the first true sampling strategy that addresses the problem of Greedy Search.

* **Intuition:** "Don't even consider the crazy, low-probability options. Only choose from a fixed-size 'shortlist' of the most likely candidates."
* **Mechanism:**
    1.  Identify the `k` tokens with the highest probabilities from the full distribution `P`.
    2.  Set the probability of all other tokens to zero.
    3.  **Re-normalize** the probabilities of the top `k` tokens so they sum to 1.
    4.  Sample from this new, smaller distribution.
* **Problem:** A fixed `k` is not adaptive. In some contexts, the number of reasonable next words might be very large (e.g., at the start of a story), and in others very small (e.g., after "The Eiffel Tower is in..."). Top-k struggles with this dynamic.

* **Code Snippet for Top-k:**
    ```python
    import torch
    import torch.nn.functional as F

    def top_k_sampling(logits, k=50):
        # logits shape: [batch_size, vocab_size]
        
        # 1. Find the top k logits and their values
        # The `topk` function returns both values and indices
        top_k_values, top_k_indices = torch.topk(logits, k) # [B, k]

        # 2. Create a new logits tensor filled with -inf
        filtered_logits = torch.full_like(logits, float('-inf'))

        # 3. Use `scatter` to place the top k values back into the filtered logits
        # This effectively sets all non-top-k logits to -inf
        filtered_logits.scatter_(1, top_k_indices, top_k_values)

        # 4. Apply softmax to get a re-normalized probability distribution
        probabilities = F.softmax(filtered_logits, dim=-1)

        # 5. Sample from this new distribution
        next_token = torch.multinomial(probabilities, num_samples=1)
        
        return next_token
    ```

#### 4. Top-p (Nucleus) Sampling

This is the most popular and effective sampling method, as it creates an adaptive shortlist.

* **Intuition:** "Instead of a fixed-size list, choose from the smallest possible list of candidates whose combined probability meets a certain threshold."
* **Mechanism:**
    1.  Sort the vocabulary tokens by their probability in descending order.
    2.  Calculate the cumulative sum of these sorted probabilities.
    3.  Find all tokens whose cumulative probability is greater than the threshold `p` (e.g., `p=0.95`). These tokens are removed from the candidate list. The remaining set is the "nucleus."
    4.  **Re-normalize** the probabilities of the nucleus tokens so they sum to 1.
    5.  Sample from this adaptive nucleus distribution.

Of course. That's an excellent request, as the tensor manipulations in Top-p sampling can be complex. Understanding the shapes at each step is key to grasping the algorithm.

Let's break down the code with detailed inline comments explaining the tensor sizes. We will assume a `batch_size` (B) of 4 and a `vocab_size` (V) of 50,000 for this example.

### Top-p (Nucleus) Sampling with Tensor Size Explanations

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.95):
    # Assume:
    B = 4  # batch_size
    V = 50000 # vocab_size
    
    # logits shape: [B, V] -> e.g., [4, 50000]
    # This is the raw output from the language model head for the last token.

    # 1. Sort logits in descending order to easily find the nucleus of high-probability tokens.
    # We need both the sorted values and their original indices to reconstruct the filter later.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # sorted_logits shape: [B, V] -> e.g., [4, 50000]
    # sorted_indices shape: [B, V] -> e.g., [4, 50000]

    # Convert sorted logits to probabilities
    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
    # sorted_probabilities shape: [B, V] -> e.g., [4, 50000]
    # Example for one row: [0.1, 0.08, 0.05, 0.02, ..., 0.00001]

    # 2. Calculate the cumulative sum of the probabilities.
    cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)
    # cumulative_probs shape: [B, V] -> e.g., [4, 50000]
    # Example for one row: [0.1, 0.18, 0.23, 0.25, ..., 1.0]

    # 3. Create a mask of tokens to remove. These are tokens that are NOT in the nucleus.
    # The nucleus is the smallest set of tokens whose cumulative probability is >= p.
    # So, we find all tokens where the cumulative probability already exceeds p.
    sorted_indices_to_remove = cumulative_probs > p
    # sorted_indices_to_remove shape: [B, V], dtype=torch.bool
    # Example for one row (if p=0.9): [False, False, ..., True, True, True]

    # 4. Shift the mask to the right to ensure we keep the first token that pushes the
    # cumulative probability over the threshold p.
    # We shift all elements one to the right, and the first element becomes False (0).
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Now the mask correctly identifies only tokens that are truly outside the nucleus.

    # 5. Go from the sorted view back to the original vocabulary order.
    # We create a boolean mask of the same shape as the original logits.
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    # We use `scatter_` to place `True` values at the original positions of the tokens we want to remove.
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    # indices_to_remove shape: [B, V], dtype=torch.bool

    # 6. Apply the mask to the original logits.
    # `masked_fill_` sets the value of logits to -inf wherever the mask is True.
    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    # filtered_logits shape: [B, V] -> e.g., [4, 50000]
    # Now contains the original logit values for nucleus tokens, and -inf for all others.

    # 7. Apply softmax to the filtered logits to get re-normalized probabilities.
    # The -inf values will become 0 after softmax, and the probabilities of the
    # nucleus tokens will be re-distributed to sum to 1.
    probabilities = F.softmax(filtered_logits, dim=-1)
    # probabilities shape: [B, V] -> e.g., [4, 50000]

    # 8. Sample one token from this new, filtered distribution.
    # `torch.multinomial` performs a weighted random draw.
    next_token = torch.multinomial(probabilities, num_samples=1)
    # next_token shape: [B, 1] -> e.g., [4, 1]

    return next_token
```

### Step-by-Step Walkthrough

Let's trace a single sequence (`B=1`) with a tiny vocabulary (`V=10`) and `p=0.9` to make it concrete.

1.  **Start with Logits:**
    `logits` = `[1.2, 3.1, 0.5, 8.2, -1.0, 5.5, 6.1, 0.1, 2.5, 4.3]` (Shape: `[1, 10]`)

2.  **Sort Logits:** We get the sorted values and their original indices.
    `sorted_logits` = `[8.2, 6.1, 5.5, 4.3, 3.1, 2.5, 1.2, 0.5, 0.1, -1.0]`
    `sorted_indices` = `[3, 6, 5, 9, 1, 8, 0, 7, 2, 4]`

3.  **Get Sorted Probabilities** (after softmax):
    `sorted_probabilities` = `[0.60, 0.18, 0.10, 0.04, 0.01, ..., ]`

4.  **Get Cumulative Probabilities:**
    `cumulative_probs` = `[0.60, 0.78, 0.88, 0.92, 0.93, ..., 1.0]`

5.  **Find Indices to Remove:** Find where `cumulative_probs > p` (where `p=0.9`).
    `sorted_indices_to_remove` (initial) = `[F, F, F, T, T, T, T, T, T, T]`

6.  **Shift the Mask:**
    `sorted_indices_to_remove` (shifted) = `[F, F, F, F, T, T, T, T, T, T]`
    *This is the key step. We keep the token that pushed us over the `p` threshold (the 4th one, with probability 0.04).* The nucleus now consists of the first 4 tokens.

7.  **Map Mask to Original Indices:** We use `sorted_indices` to put `True` (remove) at the correct original positions. We will remove all tokens *except* those at original indices `3, 6, 5, 9`.

8.  **Filter Logits:** The original `logits` tensor has the scores for tokens outside the nucleus set to `-inf`.

9.  **Re-normalize with Softmax:** We apply a final softmax. Only the 4 tokens in the nucleus will have a non-zero probability, and their probabilities will sum to 1.

10. **Sample:** `torch.multinomial` picks one token from this final 4-element distribution. The result is a single token ID.
In practice, libraries like Hugging Face `transformers` combine all these techniques into a single `.generate()` method where you can specify `temperature`, `top_k`, and `top_p` simultaneously, allowing for fine-grained control over the generation process.
---
### 7. From-Scratch Implementation of a Decoder-Only Model
This implementation uses the computationally efficient, fused-layer approach for multi-head attention.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters for our Toy Model ---
batch_size = 32      # How many independent sequences will we process in parallel?
block_size = 128     # What is the maximum context length for predictions?
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256         # Embedding dimension
n_head = 4           # Number of attention heads
n_layer = 1          # Number of decoder layers (for simplicity)
dropout = 0.2
eval_interval = 500
max_steps = 5001

# --- 1. Data Preparation and Tokenizer ---
# For this toy example, we'll use a simple character-level tokenizer.
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Warning: 'input.txt' not found. Using dummy text for demonstration.")
    text = "hello world, this is a demonstration of a toy gpt model for a top-tier graduate school class."

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Input sequences (the context)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Target sequences (the next character to predict)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. Model Components (Optimized Implementation) ---

class MultiHeadAttention(nn.Module):
    """ The efficient, fused implementation of Multi-Head Causal Self-Attention """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert n_embd % num_heads == 0
        self.head_size = n_embd // num_heads

        # A single, fused linear layer for Q, K, V projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # The final output projection layer
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        # Input x shape: [B, T, C] (Batch, Time/seq_len, Channels/n_embd)
        B, T, C = x.shape

        # 1. Fused Projection & Splitting
        qkv = self.c_attn(x) # Shape: [B, T, 3 * C]
        q, k, v = qkv.split(self.n_embd, dim=2) # Each is [B, T, C]

        # 2. Reshape for Multi-Head computation
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # 3. Batched Scaled Dot-Product Attention
        # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_size**0.5)
        att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf')) # Apply causal mask
        att = F.softmax(att, dim=-1)
        
        # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        y = att @ v

        # 4. Concatenate and Project back
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Shape: [B, T, C]
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    """ A single Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-normalization and residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 3. Full Model ---
class SimpleDecoderOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token and position embeddings
        # idx shape: [B, T]
        tok_emb = self.token_embedding_table(idx) # Shape: [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Shape: [T, C]
        x = tok_emb + pos_emb # Shape: [B, T, C]
        
        # Pass through decoder blocks
        x = self.blocks(x) # Shape: [B, T, C]
        x = self.ln_f(x) # Shape: [B, T, C]
        
        # Final projection to vocabulary size
        logits = self.lm_head(x) # Shape: [B, T, vocab_size]
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_for_loss = logits.view(B*T, C)
            targets_for_loss = targets.view(B*T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens to respect positional embedding limits
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the logit for the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- 4. Training Loop ---
model = SimpleDecoderOnlyModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for steps in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        print(f"Step {steps}, Training Loss: {loss.item():.4f}")

# --- 5. Generation from the model ---
print("\n--- Generating Text from Trained Model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = m.generate(context, max_new_tokens=200)[0].tolist()
print(decode(generated_tokens))
```
