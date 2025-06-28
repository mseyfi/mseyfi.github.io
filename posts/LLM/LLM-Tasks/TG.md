## [![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)
## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Text Generation and Completion

* **The Goal:** To produce creative, coherent, and contextually relevant text that continues from a given prompt. This is the most fundamental autoregressive task and includes applications like story writing, paraphrasing, and email completion.
* **Architecture:** The quintessential task for Decoder-Only models (like the GPT series, Llama).


#### Part 1: The Training Phase (Instruction Fine-Tuning)

**Goal:** To teach a pre-trained model to follow instructions by showing it high-quality `prompt` -> `completion` examples.


##### **Step 1.1: Prepare the Data**
You start with a dataset of prompt-completion pairs.
* **Prompt:** `"Write a short, optimistic phrase about the future."`
* **Completion:** `"The horizon ahead is bright and full of promise."`

##### **Step 1.2: The Training Process**
For each pair, we perform these steps:

1.  **Combine & Tokenize:** The prompt and completion are merged into a single sequence and converted to tokens.
    `[15, 439, 21, 74, 111, ..., 249, 13, 1146, 31, 1045, 2]`

2.  **Define Input Size (Context Window):** The model has a maximum sequence length it can handle, known as the **context window**. This is the effective "input size" during training. Common sizes are 2048, 4096, or even larger. Your combined `prompt + completion` sequence must fit within this window.

3.  **Feed Forward with a Masked Loss:**
    * The model processes the entire sequence, predicting the next token at every step.
    * **Causal Masking** ensures the model can't "cheat" by looking ahead.
    * A **Masked Cross-Entropy Loss** is applied. We only calculate the model's error on the `completion` tokens. This forces the model to learn: "Given this *kind* of prompt, generate this *kind* of completion."
---

#### Part 2: The Generation Phase (Inference)

**Goal:** Now that the model is fine-tuned, we use it to generate a new completion from a new prompt.

##### **Step 2.1: Provide a Prompt**
We start with just a prompt and convert it to tokens.
* **User Prompt:** `"Give me a tagline for a new coffee shop."`
* **Input Tokens:** `[34, 25, 14, 982, 113, 21, 53, 642, 1099, 13]`

##### **Step 2.2: The Prediction Step (Model Forward Pass)**
The model processes the input tokens to predict the *very next token*. This involves two key final steps:

1.  **The Linear Layer (Projecting to Vocabulary):** The model's final internal representation is passed to a linear layer. This layer's job is to produce a raw score (a logit) for every single token in the model's vocabulary.
    * **Output Size (Logits):** This vector's size is equal to the **vocabulary size**. If the model knows 50,000 unique tokens, this output will be a list of 50,000 numbers.
    `[-2.1, 1.5, 0.3, 8.9, -4.5, ...]` (A score for every possible word).

2.  **The Softmax Function:** The raw logit scores are hard to interpret. The **softmax** function is applied to this vector to convert the scores into a clean probability distribution.
    * **Output (Probabilities):** A vector of the same size (`vocab_size`), where each value is between 0 and 1, and all values sum to 1.
    `[0.001, 0.005, 0.002, 0.92, 0.000, ...]`
    * This tells us the model's confidence for each potential next token. In this example, the fourth token in the vocabulary is predicted with 92% probability.

##### **Step 2.3: The Sampling Strategy (Choosing the Next Word)**
We have the probabilities, now we must choose one token. Instead of always picking the most likely one, we use a sampling strategy to encourage creativity and avoid repetitive text.

* **Greedy Sampling (Not Recommended):** Always pick the token with the highest probability. **Result:** Safe but often boring and repetitive.
* **Top-K Sampling:** Consider only the `k` most likely tokens. For example, if `k=50`, you ignore all but the top 50 tokens and then sample from that smaller group based on their probabilities. **Result:** Prevents truly weird tokens from being chosen while still allowing for variety.
* **Nucleus (Top-p) Sampling (Most Common):** Consider the smallest set of tokens whose cumulative probability is greater than `p`. If `p=0.9`, you'd add up the probabilities of the most likely tokens until you hit 90%. This set might include 3 tokens if the model is very confident, or 50 tokens if it's uncertain. **Result:** An adaptive and robust strategy that produces high-quality, creative text.

##### **Step 2.4: The Autoregressive Loop**
Text generation is a loop. Let's say we sampled the token for `"Your"`.

1.  **Append:** Add the new token to our input sequence.
    * **Old Input:** `[Give me a tagline for a new coffee shop.]`
    * **New Input:** `[Give me a tagline for a new coffee shop. Your]`

2.  **Repeat:** Feed this new, longer sequence back into the model (Step 2.2). The model now predicts the next token after "Your". Maybe it's "daily".

3.  **Continue:** Repeat this process—`predict -> sample -> append`—until the model generates a special `<end_of_sequence>` token or reaches a predefined length limit.

**Final Output:** `"Your daily dose of happiness."`

***
