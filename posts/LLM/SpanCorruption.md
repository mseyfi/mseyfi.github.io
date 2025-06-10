Span corruption is a key self-supervised pre-training objective introduced in the **T5 (Text-to-Text Transfer Transformer)** model. It's designed to make the model learn to reconstruct missing contiguous spans of text, which is crucial for tasks like denoising, summarization, and machine translation.

### 1. What is Span Corruption? (Intuition)

Imagine you have a sentence, and you randomly select some words and contiguous phrases, and then replace them with a single placeholder. The model's job is to reconstruct what was originally in those placeholders. This is similar to a "fill-in-the-blanks" game, but instead of single blanks, you're filling in arbitrary-length "spans" of text.

For example, if the original sentence is:
"The quick brown fox jumps over the lazy dog."

A span corruption might turn it into:
"The quick brown $\langle X \rangle$ over the lazy dog."

The model would then be trained to predict the missing words: "fox jumps".

### 2. Is Span Corruption a Training-Only Task?

**Yes, span corruption is exclusively a pre-training task.** It's a self-supervised objective used during the initial training phase of the T5 model on a large corpus of unlabelled text. Once the model is pre-trained, it is then fine-tuned on specific downstream tasks (e.g., summarization, translation) using their respective labelled datasets and input/output formats. The mechanism of replacing spans with sentinel tokens is not used during fine-tuning or inference.

### 3. How is Span Corruption Done?

The process involves randomly selecting contiguous spans of tokens from the input sequence and replacing each selected span with a unique, special "sentinel token."

The T5 paper describes this process with two key parameters:
* **Corruption Rate (Masking Rate):** The percentage of tokens in the input sequence that will be part of a corrupted span. The paper uses 15%.
* **Mean Span Length:** The average number of tokens per corrupted span. The paper uses 3.

The actual span lengths are sampled from a Poisson distribution with $\lambda = \text{mean_span_length}$. This ensures that most spans are short, but occasionally longer spans are generated.

The process is as follows:
1.  Initialize a binary mask for the input sequence, all zeros.
2.  Randomly select a starting token for a span to be corrupted.
3.  Sample a span length $l$ from a Poisson distribution.
4.  Mark $l$ tokens starting from the selected position as corrupted (set mask to 1).
5.  Repeat until the target corruption rate is met.
6.  Each *contiguous* block of corrupted tokens (a "span") is then replaced by a unique sentinel token.

**Mathematical Formulation:**

Let $S = (t_1, t_2, \dots, t_L)$ be an input sequence of $L$ tokens.
Let $M \in \{0, 1\}^L$ be the binary mask where $M_i=1$ if $t_i$ is part of a corrupted span, and $M_i=0$ otherwise.
The total number of corrupted tokens is $\sum_{i=1}^L M_i \approx \text{corruption_rate} \cdot L$.

The corrupted input sequence $S_{enc\_in}$ is formed by concatenating non-corrupted tokens and unique sentinel tokens.
The target sequence $S_{dec\_out}$ is formed by concatenating the corrupted spans, each followed by its corresponding unique sentinel token.

### 4. What are the Training Pairs?

The training pair consists of:
* **Encoder Input:** The original sequence with corrupted spans replaced by unique sentinel tokens.
* **Decoder Output (Target):** The sequence of original corrupted spans, each delimited by its corresponding unique sentinel token, and ending with a final sentinel token (e.g., `_END_OF_SEQUENCE`).

**Example:**

Original Sequence:
`tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]`

Let's say we corrupt "fox jumps" and "lazy".

1.  **Original Spans:**
    * Span 1: `["fox", "jumps"]`
    * Span 2: `["lazy"]`

2.  **Sentinel Tokens:** T5 uses sentinel tokens like `<extra_id_0>`, `<extra_id_1>`, `<extra_id_2>`, etc. These are special tokens added to the model's vocabulary and are learnable. They are unique for each *instance* of a corrupted span within a single input, allowing the decoder to identify which span it needs to reconstruct.

3.  **Encoder Input:**
    `["The", "quick", "brown", "<extra_id_0>", "over", "the", "<extra_id_1>", "dog", "."]`

4.  **Decoder Input (shifted right during training):**
    This is derived from the target sequence by shifting it one position to the right and prepending a start-of-sequence token (or effectively, the first target token is `<extra_id_0>`).
    `["<extra_id_0>", "fox", "jumps", "<extra_id_1>", "lazy", "<extra_id_2>"]`

5.  **Decoder Output (Target):**
    `["fox", "jumps", "<extra_id_1>", "lazy", "<extra_id_2>"]`
    Notice the target sequence reconstructs the corrupted spans, each followed by the *next* sentinel token, indicating the end of that span and the start of the next. The final sentinel token `<extra_id_2>` in this example acts as an end-of-sequence marker for the target.

### 4.1 Another example with another way of implementation

🔹 Encoder Input:
Replace masked spans with sentinel tokens:

```python
["I", "enjoy", "<extra_id_0>", "in", "the", "evening", "<extra_id_1>", "lake", "."]
```
🔹 Decoder Target:
Only the missing spans, prefixed by sentinels:

```python
["<extra_id_0>", "walking", "my", "dog", "<extra_id_1>", "by", "the", "<extra_id_2>"]
```
The sentinel tokens segment the spans, and <extra_id_2> marks the end.

🔹 Decoder Input:
Shifted right version of decoder target (for teacher forcing):

```python
["<pad>", "<extra_id_0>", "walking", "my", "dog", "<extra_id_1>", "by", "the"]
```


### 5. Sentinel Tokens Explained

* **Purpose:** Sentinel tokens serve two critical roles:
    1.  **Placeholder in Encoder Input:** They mark the exact location and length of a corrupted span in the encoder input.
    2.  **Delimiter in Decoder Output:** They delimit the reconstructed spans in the decoder target. The decoder learns to predict the contents of a span until it generates the *next* sentinel token, at which point it knows to move on to the next span or to stop if it's the final sentinel.
* **Uniqueness:** Each corrupted span in a given input sequence is replaced by a *different* sentinel token (e.g., `<extra_id_0>`, `<extra_id_1>`, `<extra_id_2>`, etc.). This linkage is crucial: the decoder knows that when it generates `<extra_id_1>`, it's completing the span that was replaced by `<extra_id_0>` in the encoder input.
* **Vocabulary:** These sentinel tokens are actual tokens in the T5 vocabulary, typically occupying indices at the end of the vocabulary.


### 6. Do We Have an Extra Head for Prediction?

**No, T5 does not use an "extra head" for prediction in the sense of a separate, specialized neural network layer for span corruption.** The entire T5 model is a single encoder-decoder Transformer.

The prediction task (reconstructing the spans) is handled by the standard language modeling head which is always present in such sequence-to-sequence models: a linear layer followed by a softmax over the entire vocabulary. This layer projects the decoder's final hidden states into logits over the vocabulary, from which the next token is predicted.

The "text-to-text" paradigm of T5 means all tasks are framed as generating text. Span corruption is just one form of text generation: generating the corrupted spans.

### 7. Do We Only Care About the Sentinel Tokens in the Output?

**No, we care about predicting *all* tokens in the decoder output ($D_{out}$), including both the original content of the corrupted spans and the sentinel tokens themselves.**

The sentinel tokens in the decoder output are crucial for two reasons:
1.  They act as delimiters, indicating the end of one reconstructed span and the signal to move to the next.
2.  The model learns to generate the *correct* sentinel token (e.g., after "fox jumps", it must generate `<extra_id_1>`, not `<extra_id_0>` or some other token). This reinforces the mapping between the placeholder in the encoder input and the reconstructed content.

The loss function (e.g., cross-entropy) is computed for every token in the $D_{out}$ sequence.

### 8. Do We Create a Mask?

**Yes, masks are extensively used in the Transformer architecture, and specifically in T5's span corruption pre-training.**

There are typically two main types of masks:

1.  **Padding Mask (Attention Mask):**
    * **Purpose:** To prevent the attention mechanism from attending to padding tokens (tokens added to make sequences of equal length within a batch).
    * **Encoder:** An attention mask is created for the `Encoder Input` ($E_{in}$) to prevent self-attention from attending to padding tokens.
    * **Decoder:** An attention mask is created for the `Decoder Input` ($D_{in}$) to prevent self-attention from attending to padding tokens.
    * **Encoder-Decoder Attention:** A mask is also used to prevent the decoder from attending to padding tokens in the encoder's output.
    * **How it's done:** Typically, a binary mask is created where `1` indicates a real token and `0` indicates a padding token. During attention calculation, positions corresponding to `0` in the mask are set to a very small negative number (e.g., `-1e9`) before the softmax, effectively making their attention weights zero.

2.  **Causal Mask (Look-Ahead Mask):**
    * **Purpose:** To ensure that during decoding, a token can only attend to previous tokens in the sequence, not future ones. This maintains the auto-regressive property of text generation.
    * **Decoder Self-Attention:** This mask is applied within the decoder's self-attention layers. It's a lower triangular matrix, where the upper triangle (representing future tokens) is masked out.
    * **How it's done:** Similar to the padding mask, entries for future positions are set to a very small negative number.

**Mathematical Representation of Masking in Attention:**

Given attention scores $A_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}}$:

The masked attention scores $A'_{ij}$ are calculated as:
$$A'_{ij} = A_{ij} \text{ if } \text{mask}_{ij} = 1$$
$$A'_{ij} = -\infty \text{ if } \text{mask}_{ij} = 0$$

Then, the attention weights $P_{ij}$ are obtained via softmax:
$$P_{ij} = \text{softmax}(A'_{ij})$$
This results in $P_{ij} = 0$ for masked positions.

### 9. Code Snippets (Conceptual)

Below are conceptual PyTorch-like snippets to illustrate the process.

**a) Span Corruption Logic (Simplified)**

```python
import torch
import random
import numpy as np

# Assume these are special tokens in your vocabulary
# In T5, these would be <extra_id_0>, <extra_id_1>, etc.
SENTINEL_TOKENS = [f"<extra_id_{i}>" for i in range(100)] # Example 100 sentinels
PAD_TOKEN_ID = 0 # Example

def create_span_corruption_pair(tokens, corruption_rate=0.15, mean_span_length=3):
    """
    Creates an encoder input and decoder target for span corruption.
    tokens: list of token IDs
    """
    if not tokens:
        return [], []

    corrupted_tokens = list(tokens)
    target_spans = []
    
    # 1. Determine tokens to mask based on corruption rate
    num_tokens = len(tokens)
    num_to_mask = int(num_tokens * corruption_rate)
    
    # Create a mask to track which tokens are selected
    is_masked = np.zeros(num_tokens, dtype=bool)
    
    # Randomly select positions to start masking from
    # Ensure we don't mask too many or too few (approximate)
    masked_indices = set()
    while len(masked_indices) < num_to_mask:
        start_idx = random.randrange(num_tokens)
        if start_idx in masked_indices:
            continue
        
        # Sample span length from Poisson distribution (simplified for illustration)
        span_len = max(1, min(num_tokens - start_idx, int(np.random.poisson(mean_span_length))))
        
        # Ensure contiguous tokens in span are not already masked
        current_span_indices = []
        for i in range(start_idx, start_idx + span_len):
            if i < num_tokens and i not in masked_indices:
                current_span_indices.append(i)
            else:
                break # Stop if we hit an already masked token or end of sequence
        
        if current_span_indices:
            for idx in current_span_indices:
                is_masked[idx] = True
                masked_indices.add(idx)

    # 2. Form encoder input and decoder target
    encoder_input_ids = []
    decoder_target_ids = []
    
    current_idx = 0
    sentinel_idx_counter = 0
    
    while current_idx < num_tokens:
        if is_masked[current_idx]:
            # Found start of a masked span
            span_start = current_idx
            while current_idx < num_tokens and is_masked[current_idx]:
                current_idx += 1
            span_end = current_idx
            
            # Extract the actual span content
            actual_span_tokens = tokens[span_start:span_end]
            
            # Use a unique sentinel token for this span
            sentinel_token_id = SENTINEL_TOKENS[sentinel_idx_counter]
            encoder_input_ids.append(sentinel_token_id)
            
            # Add span content to decoder target, followed by NEXT sentinel token
            decoder_target_ids.extend(actual_span_tokens)
            sentinel_idx_counter += 1
            decoder_target_ids.append(SENTINEL_TOKENS[sentinel_idx_counter]) # Next sentinel
            
        else:
            # Not masked, add to encoder input directly
            encoder_input_ids.append(tokens[current_idx])
            current_idx += 1
            
    return encoder_input_ids, decoder_target_ids

# Example Usage:
# Assume tokenizer maps words to IDs
tokenizer = {
    "The": 1, "quick": 2, "brown": 3, "fox": 4, "jumps": 5,
    "over": 6, "the": 7, "lazy": 8, "dog": 9, ".": 10
}
# Add sentinel tokens to tokenizer
for i, s_token in enumerate(SENTINEL_TOKENS):
    tokenizer[s_token] = 11 + i # Assign IDs after regular tokens

reverse_tokenizer = {v: k for k, v in tokenizer.items()}

original_text = "The quick brown fox jumps over the lazy dog ."
original_token_ids = [tokenizer[word] for word in original_text.split()]

enc_input, dec_target = create_span_corruption_pair(original_token_ids)

print("Original Text:", original_text)
print("Encoder Input (IDs):", enc_input)
print("Encoder Input (Tokens):", [reverse_tokenizer[tid] for tid in enc_input])
print("Decoder Target (IDs):", dec_target)
print("Decoder Target (Tokens):", [reverse_tokenizer[tid] for tid in dec_target])

# To get decoder_input for training, you'd shift target:
# decoder_input_ids = [PAD_TOKEN_ID] + dec_target[:-1]
# or more commonly, in PyTorch/HuggingFace, the decoder_input_ids
# are derived by shifting the labels internally or using a start_token
# and the labels are directly dec_target.
```

**b) Masking in a Transformer Model (Conceptual PyTorch)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_padding_mask(input_ids, pad_token_id):
    """
    Creates a padding mask for attention.
    Returns: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, 1)
    """
    return (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    """
    Creates a causal (look-ahead) mask for decoder self-attention.
    Returns: (1, 1, seq_len, seq_len)
    """
    # Lower triangular matrix
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

# Inside a conceptual Attention layer's forward method:
# (Simplified, ignoring batch_size and num_heads for clarity)
class ConceptualAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # ... other attention components ...

    def forward(self, query, key, value, attention_mask=None, causal_mask=None):
        # query, key, value shape: (seq_len, d_model)

        # 1. Calculate raw attention scores
        # scores shape: (seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5)

        # 2. Apply masks
        if attention_mask is not None:
            # attention_mask: (seq_len, seq_len) or broadcastable
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        if causal_mask is not None:
            # causal_mask: (seq_len, seq_len) or broadcastable
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # 3. Softmax to get attention probabilities
        attn_probs = F.softmax(scores, dim=-1)

        # 4. Weighted sum of values
        output = torch.matmul(attn_probs, value)
        return output

# In a full Transformer model, you would combine these masks:
# For encoder self-attention: only padding mask
# For decoder self-attention: padding mask + causal mask
# For encoder-decoder attention: only padding mask from encoder output
```

Span corruption, along with the text-to-text paradigm, is a powerful pre-training strategy that allows T5 to achieve state-of-the-art results across a wide variety of NLP tasks by treating them all as a unified text generation problem.
