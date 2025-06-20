[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/FineTuning-Comprehensive_Tutorial_on_Finetuning_LLMs-orange?style=for-the-badge&logo=github)](../../FineTuning)

# A Deep Dive into P-Tuning v2
This tutorial will guide you through the concepts and architecture of **P-Tuning v2**, a landmark technique in making large language models (LLMs) adaptable and efficient. We'll explore the problem it solves, its core "deep prompt" mechanism, and why it has become a go-to method for parameter-efficient fine-tuning (PEFT).

### Part 1: The Intuition - From a Whisper to a Conversation

To understand the genius of P-Tuning v2, let's first consider the problem it solves. The original method of "prompt tuning" was a breakthrough, but it had limitations.

**Shallow Prompting (The Old Way):** Imagine an LLM is a giant, multi-story office building with many departments (the layers). The original prompt tuning was like whispering a single, cleverly-worded instruction to the intern at the front desk (the input layer). You then had to hope that single instruction was perfectly understood and relayed through every single department to guide the entire company's final decision. This "whisper" approach had two major flaws:

1.  **It was unstable:** It only worked well on gigantic models (10B+ parameters). On more common, medium-sized models, performance was poor.
2.  **It failed on hard tasks:** For complex tasks like identifying names and locations in a long sentence (sequence tagging), the single instruction wasn't nearly enough to guide the detailed, token-by-token work happening in the upper-floor departments.

**P-Tuning v2 (The New Way):** P-Tuning v2 introduces **Deep Prompt Tuning**. Instead of one instruction at the start, you place a small, expert guide on *every single floor* of the building. This guide provides continuous, layer-specific instructions to each department, ensuring the final output is exactly what you want. It's a continuous conversation with the model, not a single whisper at the start.

This deep guidance is why P-Tuning v2 is more powerful, stable across all model sizes, and excels at the complex tasks where shallow methods fail.

### Part 2: The Math & The Tensors - A Look Under the Hood

This is where we see the key architectural shift. To understand it, we must track the shape of our data as it flows through the model.

#### Key Players & Their Shapes:

  * **B**: `batch_size` (e.g., 16 examples processed at once)

  * **N**: `sequence_length` (e.g., 128 tokens in the input text)

  * **P**: `prompt_length` (a small, fixed number of virtual prompt tokens, e.g., 20)

  * **D**: `hidden_dim` (the model's internal vector size, e.g., 768 for `bert-base`)

  * **Input Embeddings** ($X\_e$): The initial text converted to vectors.

      * Shape: `[B, N, D]`

  * **Hidden States at Layer `i-1`** ($H\_{i-1}$): The output of the previous layer.

      * Shape: `[B, N, D]`

  * **Trainable Prompt for Layer `i`** ($P\_e^{(i)}$): A small, trainable matrix of prompt embeddings for a *specific layer*. This is the core of P-Tuning v2.

      * Shape: `[P, D]`

#### The P-Tuning v2 Operation at Each Layer:

The magic happens at the beginning of every transformer layer. The layer-specific prompt is prepended to the hidden states from the previous layer.

**The formula:** $H\_{i-1}' = \\text{concat}([P\_e^{(i)}; H\_{i-1}])$

Let's trace the tensor operations for **a single layer `i`**:

1.  **Get Inputs:** We start with the hidden states from the previous layer, $H\_{i-1}$.

      * $H\_{i-1}$ shape: `[B, N, D]`  (e.g., `[16, 128, 768]`)

2.  **Get the Layer's Prompt:** We retrieve the unique, trainable prompt for this specific layer, $P\_e^{(i)}$.

      * $P\_e^{(i)}$ shape: `[P, D]` (e.g., `[20, 768]`)

3.  **Prepare for Concatenation:** The prompt doesn't have a batch dimension, so we add one and expand it to match the batch size of our input text.

      * $P\_e^{(i)}$ after broadcasting: Shape becomes `[B, P, D]` (e.g., `[16, 20, 768]`)

4.  **Concatenate:** We prepend the prompt embeddings to the text embeddings along the sequence dimension.

      * `torch.cat([prompt, H_{i-1}], dim=1)`
      * Result $H\_{i-1}'$ shape: `[B, P + N, D]` (e.g., `[16, 20 + 128, 768]`)

5.  **Process through Layer:** This longer sequence is fed through the transformer layer's self-attention and feed-forward networks. The output has the same shape.

      * Output $H\_i'$ shape: `[B, P + N, D]`

6.  **Slice for Next Layer:** The prompt's job for this layer is done. We **slice it off** so the input to the next layer is back to the original sequence length, containing only the updated representations of the original tokens.

      * $H\_i = H\_i'[:, P:, :]$
      * Final output $H\_i$ shape: `[B, N, D]` (e.g., `[16, 128, 768]`)

This process—**inject, process, slice**—repeats for every single layer in the network. The only parameters updated during training are those of the prompt embeddings for all layers, ${\\theta\_{P\_e}^{(1)}, \\dots, \\theta\_{P\_e}^{(L)}}$.

### Part 3: Training Input/Output Pairs

Let's make this concrete. How does the data look for different tasks?

#### Example 1: Sentiment Classification (Sentence-Level Task)

**Goal:** Classify a movie review as "Positive" or "Negative".

  * **Input Text:** "The movie was an absolute masterpiece."
  * **Tokenized Input (`input_ids`):** A sequence of numbers representing the tokens.
  * **Target Output / Label:** `1` (representing "Positive")
  * **A single training pair would look like this:**
    ```json
    {
      "input_ids": [101, 1996, 3185, 2001, 2019, 7595, 23 masterpiece, 102],
      "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
      "label": 1
    }
    ```

#### Example 2: Named Entity Recognition (NER) (Token-Level Task)

**Goal:** Identify persons, organizations, and locations in a text. This is a "hard" task where P-Tuning v2 shines.

  * **Input Text:** "Sundar Pichai works for Google in California."
  * **Tokenized Input:** Tokens for each word.
  * **Target Output / Labels:** A label for *each token* based on the IOB (Inside, Outside, Beginning) format.
      * `["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC", "O"]`
  * **A single training pair would look like this (labels as integers):**
    ```json
    {
      "input_ids": [101, 24167, 25895, 2552, 2005, 8224, 1999, 3408, 102],
      "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],
      "labels": [1, 2, 0, 0, 3, 0, 4, 0, -100] // e.g., PER=1,2; ORG=3; LOC=4; O=0
    }
    ```
    *(Note: Often a special value like -100 is used for special tokens like [CLS] whose loss should be ignored).*

### Part 4: Inference

Once training is complete, you have a tiny set of trained prompt weights for your specific task. How do you use it to make a new prediction?

1.  **Load Models:** Load the large, frozen LLM and your small, task-specific P-Tuning v2 checkpoint (the trained `prompt_encoder`).
2.  **Prepare Input:** Take a new, unseen sentence and tokenize it just like you did for training.
3.  **Forward Pass:** Pass the tokenized input through your `P_Tuning_Model`. The model will automatically inject the learned deep prompts at every layer to guide the computation.
4.  **Get Predictions:**
      * **For Classification:** Take the final hidden state of the `[CLS]` token and pass it to your trained classification head to get the predicted label (e.g., "Positive").
      * **For NER:** Take the final hidden states of *all* the tokens and pass them to your trained token classification head to get a predicted label for each word in the sentence (e.g., "B-PER", "I-PER", etc.).

### Part 5: Code Implementation

Here is a conceptual but clean implementation using a PyTorch and Hugging Face `transformers`-style framework.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

# --- Component 1: The Trainable Prompt Encoder ---
class DeepPromptEncoder(nn.Module):
    """
    Creates a trainable prompt embedding for each layer of the LLM.
    """
    def __init__(self, num_layers, hidden_dim, prompt_length, device):
        super().__init__()
        self.prompt_length = prompt_length
        self.device = device
        
        # Create a list of embedding layers, one for each transformer layer.
        # These are the only parameters that will be trained.
        self.prompts = nn.ModuleList([
            nn.Embedding(prompt_length, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, layer_idx):
        """Returns the prompt embeddings for a specific layer index."""
        prompt_indices = torch.arange(self.prompt_length, device=self.device)
        prompt_embeddings = self.prompts[layer_idx](prompt_indices)
        return prompt_embeddings

# --- Component 2: The Main Model Wrapper ---
class P_Tuning_Model(nn.Module):
    """
    A wrapper class that injects deep prompts into a frozen base model.
    """
    def __init__(self, base_model_name, prompt_length, num_labels_list):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the base LLM and freeze all of its parameters
        self.base_model = AutoModel.from_pretrained(base_model_name).to(self.device)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Get model configuration
        self.num_layers = self.base_model.config.num_hidden_layers
        self.hidden_dim = self.base_model.config.hidden_size
        self.prompt_length = prompt_length
            
        # Instantiate our trainable prompt encoder
        self.prompt_encoder = DeepPromptEncoder(
            self.num_layers, self.hidden_dim, self.prompt_length, self.device
        ).to(self.device)
        
        # A list of classification heads for different tasks
        self.classification_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, num_labels) for num_labels in num_labels_list
        ])

    def forward(self, input_ids, attention_mask, task_idx=0):
        """
        The main forward pass that injects prompts at each layer.
        """
        batch_size = input_ids.shape[0]
        
        # Get initial word embeddings
        hidden_states = self.base_model.embeddings(input_ids=input_ids)

        # Loop through each transformer layer
        for i, layer in enumerate(self.base_model.encoder.layer):
            prompt = self.prompt_encoder(layer_idx=i).unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = torch.cat([prompt, hidden_states], dim=1)
            
            prompt_attention_mask = torch.ones(batch_size, self.prompt_length, device=self.device)
            extended_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            
            # The transformers library internally creates the 4D attention mask
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask
            )
            hidden_states = layer_outputs[0]
            hidden_states = hidden_states[:, self.prompt_length:, :]

        # Use the appropriate classification head for the task
        # For sequence classification, use the [CLS] token's output
        cls_output = hidden_states[:, 0]
        logits = self.classification_heads[task_idx](cls_output)
        
        return logits

```

### Part 6: Reference

The work described in this tutorial is based on the following paper:

  * Liu, X., Ji, K., Fu, Y., Tam, W. L., Du, Z., Yang, Z., & Tang, J. (2022). **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.** *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*.
