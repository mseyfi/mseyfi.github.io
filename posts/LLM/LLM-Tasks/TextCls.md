[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Text Classification, Sentiment Analysis

Use an LLM to perform text classification—the task of assigning a predefined category (like `Positive`, `Negative`, `Spam`, or `Legal`) to a piece of text. We will cover the two primary architectural approaches.

---

#### Approach 1: The Encoder-Only Model (e.g., BERT)

This is the traditional and highly efficient method, optimized specifically for classification and understanding tasks.

##### **Part 1: The Training Phase** ⚙️

**Goal:** To train a model to "read" an entire piece of text and map its meaning to a single categorical label.

1.  **Prepare the Data:** The data consists of simple `(text, label)` pairs.
    * **Text:** `"I love this product! It works perfectly."`
    * **Label:** `Positive` (often represented as an integer, e.g., 0 for Positive, 1 for Negative).

2.  **The Training Process:**
    * **Input Formatting & Special Tokens:** Each text input is formatted with special tokens. A `[CLS]` (classification) token is added to the beginning, and a `[SEP]` (separator) token is added to the end.
        `[CLS] I love this product! It works perfectly. [SEP]`
    * **Input Size:** The input is truncated or padded to match the model's fixed **context window**, which is typically 512 tokens for models like BERT.
    * **Architecture & Masking:** The key is that encoder-only models are **bi-directional**. They use an **Attention Mask** to ignore padding tokens but do *not* use a causal mask. This allows every token to "see" every other token in the text, providing a holistic understanding.
    * **Loss Function:** The model processes the text. The final hidden state corresponding to the `[CLS]` token is used as a summary vector for the entire sequence. This vector is fed into a simple "classification head" (a linear layer) whose output size is the **number of classes**. If you have 3 labels (Positive, Negative, Neutral), the output size is 3. A standard **Cross-Entropy Loss** is calculated between the predicted class probabilities and the true label.

##### **Part 2: The Inference Phase** 💡

1.  **Format the Input:** A new, unseen piece of text is formatted with `[CLS]` and `[SEP]`.
    `[CLS] The flight was delayed again, and the staff was unhelpful. [SEP]`

2.  **Prediction:** The model performs a forward pass. It extracts the final embedding of the `[CLS]` token and passes it to the trained classification head. A **softmax** function is applied to the head's output to get a probability for each class.
    * **Output:** `[0.01, 0.99]` (Probabilities for Positive, Negative)

3.  **Final Label:** The label with the highest probability is chosen as the result.
    * **Result:** `Negative`
---

#### Approach 2: The Decoder-Only Model (e.g., GPT, Llama)

This modern approach cleverly reframes classification as a text generation task.

##### **Part 1: The Training Phase** ✍️

**Goal:** To teach the model to complete a prompt with the correct label word.

1.  **Prepare the Data:** The `(text, label)` pairs are converted into a prompt-completion format.
    * **Prompt:** `"Review: I love this product! It works perfectly. Sentiment:"`
    * **Completion:** `Positive`

2.  **The Training Process:**
    * **Input Formatting:** The prompt and completion are combined into a single sequence for next-token prediction.
        `Review: I love this product! It works perfectly. Sentiment: Positive`
    * **Input Size:** The sequence must fit within the model's **context window** (e.g., 4096 tokens).
    * **Architecture & Masking:** This is a standard **decoder-only** setup using **causal masking**. The model can only see past tokens when predicting the next one.
    * **Loss Function:** A **Masked Cross-Entropy Loss** is used. The loss is calculated **only on the token(s) for the target label**. For instance, the model is only graded on its ability to predict the token `Positive` after seeing `... Sentiment:`.

##### **Part 2: The Inference Phase** 🔍

1.  **Format the Input:** The new text is placed into the prompt template, leaving the answer blank for the model to fill in.
    `Review: The flight was delayed again, and the staff was unhelpful. Sentiment:`

2.  **Prediction:** The model runs a forward pass and generates probabilities for the very next token. We are specifically interested in the probabilities of our potential label tokens (`Positive`, `Negative`, `Neutral`).
    * The model will output a probability distribution over the entire vocabulary.
    * We check the probabilities for our specific target words: `P("Positive") = 0.01`, `P("Negative") = 0.99`, `P("Neutral") = 0.00`.

3.  **Final Label:** We select the valid label token with the highest probability. This method ensures the model gives a constrained and valid answer.
    * **Result:** `Negative`

***
