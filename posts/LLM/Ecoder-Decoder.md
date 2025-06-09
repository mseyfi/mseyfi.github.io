## A Deep Dive into Encoder-Decoder Transformers (Definitive Edition)

*Last Updated: June 9, 2025*

### 1\. The Task: Sequence-to-Sequence (Seq2Seq) Transformation 📝

Encoder-decoder models are the canonical choice for any task that maps one sequence of arbitrary length to another. They form a powerful and flexible framework for problems where the input and output have different lengths, structures, or even languages.

  * **Classic Examples:**
      * **Machine Translation:** Translating a sentence from English (input sequence) to German (output sequence).
      * **Summarization:** Condensing a long article (input sequence) into a short paragraph (output sequence).
      * **Conversational Response Generation:** Taking a user's question (input sequence) and generating a helpful answer (output sequence).

The **T5 (Text-to-Text Transfer Transformer)** model unified many NLP tasks under this single, powerful framework by framing every problem as a text-to-text conversion.

-----

![image](../../images/VIT.png)

*Fig. 1: A common Encoder decoder architecture in LLMs*

### 2\. The Encoder-Decoder Architecture 🏛️

This architecture consists of two distinct Transformer stacks connected by a specific attention mechanism.

#### 2.1 The Encoder Stack

  * **Function:** To "understand" and "encode" the input sequence.
  * **Mechanism:** A stack of Transformer encoder layers. Each layer contains a **bidirectional self-attention** mechanism, allowing every token in the input to attend to every other token. This creates a rich, contextualized representation of the entire input sequence.
  * **Output:** A sequence of hidden state vectors, `encoder_outputs`, one for each input token, representing the context-aware meaning of those tokens.

#### 2.2 The Decoder Stack

  * **Function:** To "generate" the output sequence autoregressively (one token at a time).
  * **Mechanism:** A stack of Transformer decoder layers. The decoder is more complex and has two types of attention layers:
    1.  **Masked Self-Attention:** This is the same causal self-attention we saw in decoder-only models. It allows each token being generated to attend to the *previous* tokens in the generated output sequence.
    2.  **Cross-Attention:** This is the crucial link between the encoder and decoder.

-----

### 3\. Key Mathematical Concepts in the Architecture 🧠

#### 3.1 Positional Encoding in the Encoder and Decoder

Because the self-attention mechanism is permutation-invariant, we must inject information about the order of tokens. In the standard Transformer, this is done via an additive positional encoding ($PE$).

  * **In the Encoder:** The positional encoding is added to the input token embeddings at the bottom of the encoder stack. For an input sequence $X = (x\_1, ..., x\_n)$, the input to the first layer is $X'\_{pos}$:
    $$x'_{pos, i} = x_{token, i} + PE_i$$

  * **In the Decoder:** The exact same process is applied independently to the decoder's inputs. For the target sequence $Y = (y\_1, ..., y\_m)$ being generated, the input to the decoder's self-attention at step `t` is based on the embedding of the previously generated token, $y\_{t-1}$, plus its positional encoding, $PE\_{t-1}$:
    $$y'_{pos, t-1} = y_{token, t-1} + PE_{t-1}$$

#### 3.2 Causal Masking in the Decoder

  * **Purpose:** To preserve the autoregressive property of the decoder. When predicting the token at position `t`, the decoder must only be allowed to see the tokens at positions `< t`.
  * **Application:** Causal masking is applied **only within the decoder's first attention block (the masked self-attention layer)**. It is **not** used in the encoder (which must be bidirectional) and it is **not** used in the cross-attention layer.
  * **Mathematical Formulation:** The causal mask is a matrix, $M\_{causal}$, that is added to the attention scores before the softmax.
    $$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$
    The masked self-attention calculation in the decoder is therefore:
    $$\text{MaskedSelfAttention}(Q_d, K_d, V_d) = \text{softmax}\left(\frac{Q_d K_d^T}{\sqrt{d_k}} + M_{causal}\right)V_d$$
    where $Q\_d, K\_d, V\_d$ are all derived from the decoder's own input sequence. The `-∞` values become zero after the softmax, preventing any "flow" of information from future tokens.

#### 3.3 Cross-Attention: The Mathematical Bridge

Cross-attention is the mechanism that allows the decoder to "look at" the input sequence to inform its generation process. It occurs in the second attention block of every decoder layer.

  * **Mechanism:**
    1.  The **Query (Q)** vectors are generated from the output of the decoder's masked self-attention sub-layer below it. Let's call this intermediate decoder representation $H\_d$.
        $$Q_d = H_d W^Q_{cross}$$
    2.  The **Key (K) and Value (V)** vectors are generated from the **final output of the encoder stack**, `encoder_outputs`. These vectors are created once and are used in every decoder layer.
        $$K_e = \text{encoder\_outputs} \cdot W^K_{cross}$$     $$V_e = \text{encoder\_outputs} \cdot W^V_{cross}$$
  * **Mathematical Formulation:** The cross-attention calculation is a standard attention formula, but with inputs from different sources. **Crucially, no causal mask is applied here.** The decoder is allowed to attend to *all* parts of the input sentence at every step.
    $$\text{CrossAttention}(Q_d, K_e, V_e) = \text{softmax}\left(\frac{Q_d K_e^T}{\sqrt{d_k}}\right)V_e$$
    This operation produces a context vector that is a weighted sum of the encoder's output values, where the weights are determined by how relevant each input token is to the decoder's current generation step.

-----

### 4\. T5 Architecture by the Numbers 🔢

The T5 model was released in several sizes. The core architecture remains the same, but the depth, width, and number of attention heads are scaled.

| Model Variant | Total Parameters | Num. Layers (Encoder/Decoder) | Hidden Size ($d\_{model}$) | Feed-Forward Size ($d\_{ff}$) | Num. Heads | Head Size ($d\_{kv}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T5-Small** | 60 Million | 6 / 6 | 512 | 2048 | 8 | 64 |
| **T5-Base** | 220 Million | 12 / 12 | 768 | 3072 | 12 | 64 |
| **T5-Large** | 770 Million | 24 / 24 | 1024 | 4096 | 16 | 64 |
| **T5-3B** | 3 Billion | 24 / 24 | 1024 | 16384 | 32 | 128 |
| **T5-11B** | 11 Billion | 24 / 24 | 1024 | 65536 | 128 | 128 |

-----

### 5\. The Final Output Layer and Training ⚙️

#### 5.1 Final Output Layer

The final output layer exists at the top of the **decoder stack**. It is a single linear layer that takes the final hidden state vector from the top decoder layer at a given timestep `t` and projects it to the size of the vocabulary (`[hidden_size, vocab_size]`). A `softmax` function is then applied to these logits to get a probability distribution for the next token.

#### 5.2 Input-Output Training Pairs

All tasks are framed as text-to-text. The input is a text string, often with a task-specific prefix, and the output is the target text string.

  * **Pre-training (Denoising Objective):**
      * **Input:** A corrupted sentence. e.g., `Thank you <X> me to your party <Y> week.`
      * **Output (Label):** The sentinel tokens and the text they replaced. e.g., `<X> for inviting <Y> last <Z>`
  * **Fine-tuning (Summarization Task):**
      * **Input:** `summarize: <long article text>`
      * **Output (Label):** `<short summary text>`

#### 5.3 Loss Function and Metrics

  * **Loss Function:** The model is trained to minimize the **Cross-Entropy Loss** between the decoder's predicted output and the true target sequence.
    $$L(\theta) = - \sum_{t=1}^{m} \log P(y_t | y_{<t}, X; \theta)$$
  * **Metrics:** For seq2seq tasks, we often use metrics like **BLEU** (for translation) and **ROUGE** (for summarization) to compare the generated output against a human-written reference.

-----

### 6\. From-Scratch Implementation of an Encoder-Decoder Model 💻

This section provides a simple, end-to-end implementation of an Encoder-Decoder Transformer using PyTorch's fundamental building blocks. This code prioritizes clarity to demonstrate the fundamental data flow.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters for our Toy Model ---
batch_size = 32
block_size = 128 # Max sequence length for both source and target
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256 # Embedding dimension
n_head = 4   # Number of attention heads
n_layer = 2  # Number of encoder and decoder layers
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
    text = "This is a simple demonstration of an encoder-decoder model. The task will be to 'summarize' a short text by learning to replicate its first half."

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Create a simple seq2seq task: given the first half of a block, predict the second half
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    src = torch.stack([data[i:i+block_size//2] for i in ix])
    tgt = torch.stack([data[i+block_size//2:i+block_size] for i in ix])
    return src.to(device), tgt.to(device)

# --- 2. Model Components ---

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention (Can be causal or non-causal) """
    def __init__(self, n_embd, num_heads, is_causal=False):
        super().__init__()
        self.num_heads, self.head_size = num_heads, n_embd // num_heads
        assert n_embd % num_heads == 0
        # Using a fused layer for Q, K, V for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        if self.is_causal:
            # Causal mask to ensure attention is only applied to the left in the decoder.
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, context=None):
        B, T, C = x.shape
        # If context is provided, this is cross-attention. Q is from x, K and V are from context.
        if context is not None:
            q = self.c_attn(x).split(self.n_embd, dim=2)[0]
            k, v = self.c_attn(context).split(self.n_embd, dim=2)[1:]
        else: # This is self-attention. Q, K, and V are all from x.
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        T_ctx = context.shape[1] if context is not None else T
        k = k.view(B, T_ctx, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T_ctx, self.num_heads, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_size**0.5)
        if self.is_causal:
            att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    """ Encoder Block: Bidirectional self-attention followed by FF """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, is_causal=False)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    """ Decoder Block: Causal self-attention, cross-attention, then FF """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, is_causal=True)
        self.cross_attn = MultiHeadAttention(n_embd, n_head, is_causal=False)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x, context):
        # x is the target sequence from the decoder
        # context is the output from the encoder
        x = x + self.sa(self.ln1(x)) # Causal self-attention on decoder's own output
        x = x + self.cross_attn(self.ln2(x), context=context) # Cross-attention with encoder output
        x = x + self.ffwd(self.ln3(x))
        return x

# --- 3. Full Encoder-Decoder Model ---
class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.encoder = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # We must implement the decoder as a module list to handle the context passing
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, src, tgt, targets=None):
        B_src, T_src = src.shape
        B_tgt, T_tgt = tgt.shape
        
        # --- Encoder Pass ---
        src_tok_emb = self.token_embedding_table(src) # (B, T_src, C)
        src_pos_emb = self.position_embedding_table(torch.arange(T_src, device=device)) # (T_src, C)
        src_x = src_tok_emb + src_pos_emb
        encoder_output = self.encoder(src_x) # (B, T_src, C)

        # --- Decoder Pass ---
        tgt_tok_emb = self.token_embedding_table(tgt) # (B, T_tgt, C)
        tgt_pos_emb = self.position_embedding_table(torch.arange(T_tgt, device=device)) # (T_tgt, C)
        decoder_x = tgt_tok_emb + tgt_pos_emb
        
        # Manually pass through decoder blocks with context
        for block in self.decoder_blocks:
            decoder_x = block(decoder_x, context=encoder_output)

        logits = self.lm_head(self.ln_f(decoder_x)) # (B, T_tgt, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, src, max_new_tokens):
        self.eval()
        src = src.to(device)
        # Start with a beginning-of-sequence token (0) for the decoder
        tgt = torch.zeros((src.shape[0], 1), dtype=torch.long, device=device)

        # Encoder pass is done only once
        src_tok_emb = self.token_embedding_table(src)
        src_pos_emb = self.position_embedding_table(torch.arange(src.shape[1], device=device))
        encoder_output = self.encoder(src_tok_emb + src_pos_emb)
        
        for _ in range(max_new_tokens):
            # Decoder pass in a loop
            tgt_cond = tgt[:, -block_size:] # Ensure context doesn't exceed block size
            
            tgt_tok_emb = self.token_embedding_table(tgt_cond)
            tgt_pos_emb = self.position_embedding_table(torch.arange(tgt_cond.shape[1], device=device))
            decoder_x = tgt_tok_emb + tgt_pos_emb
            
            for block in self.decoder_blocks:
                decoder_x = block(decoder_x, context=encoder_output)
            
            logits = self.lm_head(self.ln_f(decoder_x))
            
            # Focus on the last token's prediction
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tgt = torch.cat((tgt, next_token), dim=1)
        self.train()
        return tgt

# --- 4. Training Loop ---
model = EncoderDecoderModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for steps in range(max_steps):
    src, tgt = get_batch('train')
    # For training, the target `y` is the same as the decoder input `tgt`.
    # The loss function handles the shifting automatically.
    logits, loss = model(src, tgt, targets=tgt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        print(f"Step {steps}, Training Loss: {loss.item():.4f}")

# --- 5. Inference from the model ---
print("\n--- Generating Text from Trained Model ---")
src, _ = get_batch('val')
src_text_full = decode(src[0].tolist())
# We need to provide the same source sequence to the generate function
src_for_gen = src[:1, :] # Use the first sequence from the validation batch

generated_tokens = m.generate(src_for_gen, max_new_tokens=src.shape[1])[0].tolist()
generated_text = decode(generated_tokens)

print(f"Source Text:\n'{src_text_full}'")
print(f"\nGenerated Text:\n'{generated_text}'")
```
