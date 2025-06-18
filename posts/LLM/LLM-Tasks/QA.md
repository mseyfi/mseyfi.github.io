[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Question Answering (Q & A)

#### 5a. Extractive Question Answering

##### **Example Data**
The data consists of a `Context` and a `Question`. The target is not a string of text, but rather the starting and ending positions (indices) of the answer within the context.

* **Context:** `"The first iPhone was introduced by Apple CEO Steve Jobs on January 9, 2007, at the Macworld convention."`
* **Question:** `"Who introduced the first iPhone?"`
* **Target:** `start_token_index: 10`, `end_token_index: 11` (Assuming "Steve" is the 10th token and "Jobs" is the 11th).

##### **Use Case Scenario**
The goal is to find and highlight a precise answer that already exists within a provided document. This is ideal for systems that need to be factual and grounded in a specific text.

* **A legal assistant is reviewing a contract:** The user asks, `"What is the termination clause notice period?"` The system scans the contract (`Context`) and extracts the exact phrase, like `"thirty (30) business days"`, highlighting it for the user.
* **Corporate knowledge base:** An employee asks the HR portal, `"What is the maximum rollover for vacation days?"` The system finds the answer directly in the employee handbook.
---

##### **How It Works: A Mini-Tutorial (Encoder-Only)**
This task is perfectly suited for Encoder-Only models like BERT because they can analyze the relationship between the question and the entire context simultaneously.

##### **Why Encoders Dominate Extractive QA**
The model needs to understand which part of the `Context` is most relevant to the `Question`. A bi-directional encoder can create rich connections between words in the question (e.g., "Who") and words in the context (e.g., "Steve Jobs"), making it highly effective at identifying the precise answer span.

##### **The Training Phase **

1.  **Input Formatting:** The question and context are combined into a single sequence for the model, separated by a special `[SEP]` token.
    `[CLS] Who introduced the first iPhone? [SEP] The first iPhone was introduced by Apple CEO Steve Jobs... [SEP]`
2.  **Architecture & The "Two Heads":** This is the unique part. Instead of one classification head on the `[CLS]` token, we add **two separate prediction heads** on top of **every single token's output vector** in the sequence.
    * **Start Token Head:** A simple linear layer that takes each token's final output vector (e.g., size 768) and produces a single score (a logit). This score represents how likely that token is to be the *start* of the answer.
    * **End Token Head:** A *different* linear layer that does the same thing, but its score represents how likely that token is to be the *end* of the answer.
3.  **Loss Function:** After a forward pass, we have two lists of scores: one for all possible start positions and one for all possible end positions.
    * A **softmax** function is applied across each list to convert the scores into probabilities.
    * The total loss is the sum of two **Cross-Entropy Losses**:
        1.  The loss for the start position (comparing the start probabilities to the true `start_token_index`).
        2.  The loss for the end position (comparing the end probabilities to the true `end_token_index`).
    * This trains the model to jointly predict the correct start and end boundaries for the answer.

##### **The Inference Phase 💡**

1.  **Format and Predict:** A new `(Question, Context)` pair is formatted and passed through the fine-tuned model. This produces two lists of probabilities: one for the start position and one for the end position.
2.  **Find Best Span:** We find the token index with the highest start probability (`best_start`) and the token index with the highest end probability (`best_end`).
3.  **Post-Process:** We check for valid spans (e.g., ensuring `best_end >= best_start`). More advanced methods find the span `(i, j)` that maximizes `P_start(i) * P_end(j)`.
4.  **Extract Answer:** The tokens from the `best_start` index to the `best_end` index are selected from the original context and converted back to a string. This string is the final answer.

---


#### 1. How do we ensure `end_token_index >= start_token_index`?

You are absolutely right to ask this. If you simply take the `argmax` (the index with the highest score) of the start probabilities and the `argmax` of the end probabilities independently, you could easily end up with a `start_index` of 25 and an `end_index` of 10, which is nonsensical.

This is handled during the **inference/post-processing stage**. Here are the common strategies, from simple to robust:

##### Simple Approach: Check After Predicting
1.  Find the index of the highest start probability, let's call it `best_start_idx`.
2.  Find the index of the highest end probability, let's call it `best_end_idx`.
3.  **Check the condition:** If `best_end_idx < best_start_idx`, you conclude that the model could not find a valid answer and return an empty string or a "no answer found" message.

* **Drawback:** This is a bit naive. The model might have predicted a slightly lower-scoring but *valid* span (e.g., the second-best start and second-best end form a valid pair) that this approach would miss.

##### Robust Approach: Find the Best *Valid* Span (Most Common)

This is the standard and correct way to do it. Instead of finding the best start and end independently, you search for the **pair `(start, end)` that has the highest combined score, while respecting the constraint.**

Here is the algorithm:

1.  After the model's forward pass, you have two lists of scores (logits): `start_scores` and `end_scores`.
2.  Initialize a `max_score = -infinity` and `best_span = (null, null)`.
3.  Iterate through every possible start token index `i` in the sequence.
4.  For each `i`, iterate through every possible end token index `j`, **starting from `i`** (`j >= i`).
5.  You can also add another constraint to limit the maximum answer length, e.g., `if (j - i + 1 > max_answer_length): continue`.
6.  Calculate the combined score for the span `(i, j)`. The simplest way is `score = start_scores[i] + end_scores[j]`.
7.  If this `score` is greater than the current `max_score`, update:
    * `max_score = score`
    * `best_span = (i, j)`
8.  After checking all possible valid pairs, `best_span` will hold the indices for the best possible valid answer according to the model.

This search guarantees that your final `start_index` and `end_index` form a valid, logical span.

***

#### 2. Do we take the softmax over the entire context length?

**Yes, absolutely.** You take the softmax over the **entire input sequence length**.

Let's clarify why.

1.  **Input:** Your model receives a single, long sequence of tokens: `[CLS] question_tokens [SEP] context_tokens [SEP]`. Let's say the total length is 384 tokens.
2.  **Output from Heads:** The model produces two vectors of raw scores (logits), each of length 384.
    * `start_logits` = `[score_for_token_0, score_for_token_1, ..., score_for_token_383]`
    * `end_logits` = `[score_for_token_0, score_for_token_1, ..., score_for_token_383]`
3.  **Softmax Application:** The softmax function is applied independently to each of these full vectors.
    * `start_probabilities = softmax(start_logits)`
    * `end_probabilities = softmax(end_logits)`

The result is two probability distributions, each of length 384, where the probabilities in each list sum to 1.

**Why is this necessary?**

The task is to find the single best start token **out of all possible tokens** in the context. The softmax function creates a "competition" among all the tokens. By applying it to the entire sequence, you are forcing the model to decide which token, from position 0 to 383, is the most likely start. The same logic applies to the end token.

If you only applied it to a smaller chunk, the model wouldn't be able to compare a potential answer at the beginning of the context with one at the end. The full-sequence softmax is what allows the model to pinpoint the single most probable start and end point across the entire provided text.

---

#### 5b. Generative (Abstractive) Question Answering

##### **Example Data**
The data is typically a `Question` and a free-form `Answer`. The answer is not necessarily a direct quote from any text.

* **Input (Question):** `"Why is the sky blue?"`
* **Target (Answer):** `"The sky appears blue because of a phenomenon called Rayleigh scattering, where shorter-wavelength light, like blue and violet, is scattered more effectively by the tiny molecules of air in Earth's atmosphere."`

##### **Use Case Scenario**
The goal is to generate a natural, human-like answer, either by synthesizing information from a provided document (abstractive summarization) or by drawing from its own vast internal knowledge learned during pre-training. This is the foundation of modern conversational AI.

* **General knowledge queries:** A student uses Google or ChatGPT to ask, `"Explain the process of photosynthesis in simple terms."`
* **Customer support chatbot:** A user asks, `"How do I reset my account password?"` The chatbot generates a step-by-step guide instead of just pointing to a paragraph in a manual.
---
##### **How It Works: A Mini-Tutorial (Decoder-Only or Encoder-Decoder)**
This is a classic text generation task, perfectly suited for Decoder-Only models like GPT or Llama.

##### **Why Decoders Excel at Generative QA**
The task is not to *find* an answer, but to *create* one. This requires generating a novel sequence of words in a coherent, fluent manner. This is precisely what the autoregressive, next-token-prediction nature of decoder models is designed to do.

##### **The Training Phase ✍️**

1.  **Input Formatting:** The training data is structured as a prompt-completion task. The prompt contains the question (and optionally a context), and the completion is the desired answer.
    * **Format:** `Question: Why is the sky blue? Answer:`
2.  **Architecture & Loss:** The setup is identical to other text generation tasks.
    * The model is a standard **decoder-only** architecture.
    * It uses **Causal Masking** to ensure it only sees past tokens when predicting the next one.
    * The loss function is **Masked Cross-Entropy Loss**, where the error is calculated *only* on the tokens of the target answer. This teaches the model: "When you see this question, generate this answer."

##### **The Inference Phase 🗣️**

1.  **Format Prompt:** The user's question is formatted into the same template used during training (e.g., `Question: Why is the sky blue? Answer:`).
2.  **Generate Autoregressively:** The model begins generating the answer one token at a time.
3.  **The Loop:** At each step, the model predicts the probabilities for the next token using its **linear layer** and a **softmax** function. A token is chosen via a **sampling strategy** (like Top-p). This new token is appended to the sequence, and the process repeats.
4.  **Complete Answer:** The generation loop continues until the model produces a special end-of-sequence token or reaches a predefined length limit. The full sequence of generated tokens is the final answer.

---


