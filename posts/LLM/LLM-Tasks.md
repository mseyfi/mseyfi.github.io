[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)



# A Comprehensive Guide to LLM Tasks: From Fine-Tuning to Advanced Applications

A single, powerful pre-trained Large Language Model is a versatile foundation that can be adapted—or "fine-tuned"—to excel at a wide array of specific tasks. The key to this versatility lies in how we frame the task and format the data.

This tutorial provides a detailed breakdown of the most common and important tasks an LLM can perform, explaining for each: its goal, the data format for fine-tuning, a practical example, and the underlying mechanics.

***
## ![GenAI](../../badges/text_completion.svg?style=flat&logo=homeadvisor&logoColor=white)

* **The Goal:** To produce creative, coherent, and contextually relevant text that continues from a given prompt. This is the most fundamental autoregressive task and includes applications like story writing, paraphrasing, and email completion.
* **Architecture:** The quintessential task for Decoder-Only models (like the GPT series, Llama).


### Part 1: The Training Phase (Instruction Fine-Tuning)

**Goal:** To teach a pre-trained model to follow instructions by showing it high-quality `prompt` -> `completion` examples.


#### **Step 1.1: Prepare the Data**
You start with a dataset of prompt-completion pairs.
* **Prompt:** `"Write a short, optimistic phrase about the future."`
* **Completion:** `"The horizon ahead is bright and full of promise."`

#### **Step 1.2: The Training Process**
For each pair, we perform these steps:

1.  **Combine & Tokenize:** The prompt and completion are merged into a single sequence and converted to tokens.
    `[15, 439, 21, 74, 111, ..., 249, 13, 1146, 31, 1045, 2]`

2.  **Define Input Size (Context Window):** The model has a maximum sequence length it can handle, known as the **context window**. This is the effective "input size" during training. Common sizes are 2048, 4096, or even larger. Your combined `prompt + completion` sequence must fit within this window.

3.  **Feed Forward with a Masked Loss:**
    * The model processes the entire sequence, predicting the next token at every step.
    * **Causal Masking** ensures the model can't "cheat" by looking ahead.
    * A **Masked Cross-Entropy Loss** is applied. We only calculate the model's error on the `completion` tokens. This forces the model to learn: "Given this *kind* of prompt, generate this *kind* of completion."
#

### Part 2: The Generation Phase (Inference)

**Goal:** Now that the model is fine-tuned, we use it to generate a new completion from a new prompt.

#### **Step 2.1: Provide a Prompt**
We start with just a prompt and convert it to tokens.
* **User Prompt:** `"Give me a tagline for a new coffee shop."`
* **Input Tokens:** `[34, 25, 14, 982, 113, 21, 53, 642, 1099, 13]`

#### **Step 2.2: The Prediction Step (Model Forward Pass)**
The model processes the input tokens to predict the *very next token*. This involves two key final steps:

1.  **The Linear Layer (Projecting to Vocabulary):** The model's final internal representation is passed to a linear layer. This layer's job is to produce a raw score (a logit) for every single token in the model's vocabulary.
    * **Output Size (Logits):** This vector's size is equal to the **vocabulary size**. If the model knows 50,000 unique tokens, this output will be a list of 50,000 numbers.
    `[-2.1, 1.5, 0.3, 8.9, -4.5, ...]` (A score for every possible word).

2.  **The Softmax Function:** The raw logit scores are hard to interpret. The **softmax** function is applied to this vector to convert the scores into a clean probability distribution.
    * **Output (Probabilities):** A vector of the same size (`vocab_size`), where each value is between 0 and 1, and all values sum to 1.
    `[0.001, 0.005, 0.002, 0.92, 0.000, ...]`
    * This tells us the model's confidence for each potential next token. In this example, the fourth token in the vocabulary is predicted with 92% probability.

#### **Step 2.3: The Sampling Strategy (Choosing the Next Word)**
We have the probabilities, now we must choose one token. Instead of always picking the most likely one, we use a sampling strategy to encourage creativity and avoid repetitive text.

* **Greedy Sampling (Not Recommended):** Always pick the token with the highest probability. **Result:** Safe but often boring and repetitive.
* **Top-K Sampling:** Consider only the `k` most likely tokens. For example, if `k=50`, you ignore all but the top 50 tokens and then sample from that smaller group based on their probabilities. **Result:** Prevents truly weird tokens from being chosen while still allowing for variety.
* **Nucleus (Top-p) Sampling (Most Common):** Consider the smallest set of tokens whose cumulative probability is greater than `p`. If `p=0.9`, you'd add up the probabilities of the most likely tokens until you hit 90%. This set might include 3 tokens if the model is very confident, or 50 tokens if it's uncertain. **Result:** An adaptive and robust strategy that produces high-quality, creative text.

#### **Step 2.4: The Autoregressive Loop**
Text generation is a loop. Let's say we sampled the token for `"Your"`.

1.  **Append:** Add the new token to our input sequence.
    * **Old Input:** `[Give me a tagline for a new coffee shop.]`
    * **New Input:** `[Give me a tagline for a new coffee shop. Your]`

2.  **Repeat:** Feed this new, longer sequence back into the model (Step 2.2). The model now predicts the next token after "Your". Maybe it's "daily".

3.  **Continue:** Repeat this process—`predict -> sample -> append`—until the model generates a special `<end_of_sequence>` token or reaches a predefined length limit.

**Final Output:** `"Your daily dose of happiness."`

***

## ![GenAI](../../badges/dialogue_generation.svg)

This tutorial explains how to take a general-purpose LLM and fine-tune it to be an interactive, multi-turn conversational agent that can remember context and adopt a specific persona.
#

### Part 1: The Training Phase (Fine-Tuning for Dialogue)

**Goal:** To teach a pre-trained model the structure, flow, and turn-taking nature of a human conversation. The model learns to act as a helpful assistant.

#### **Step 1.1: Prepare the Data (The Conversational Script)**

You start with a dataset of multi-turn dialogues. These dialogues are formatted with **special tokens** to clearly define who is speaking. This structure is critical for the model to learn its role.

Let's look at the format from your example (`<s>` is start of sequence, `</s>` is end of sequence, `[INST]` marks a user instruction).

**Example Dialogue:**
```
<s>[INST] What’s the capital of Germany? [/INST] The capital of Germany is Berlin.</s><s>[INST] What about France? [/INST] The capital of France is Paris.</s>
```

* **Special Tokens:** `<s>`, `[INST]`, `[/INST]`, `</s>` are not words; they are structural signposts. They teach the model: "When you see text between `[INST]` and `[/INST]`, that's the user speaking. Your job is to provide the text that comes after `[/INST]`."

#### **Step 1.2: The Training Process**

1.  **Combine & Tokenize:** The entire conversation, including all turns and special tokens, is concatenated into one long sequence and converted into numerical tokens.

2.  **Define Input Size (Context Window):** The model's context window (e.g., 4096 tokens) is crucial. A longer context window allows the chatbot to "remember" more of the previous conversation, leading to more coherent and context-aware responses.

3.  **Feed Forward with a Role-Specific Masked Loss:**
    * **Causal Masking** is always active. The model can only see past tokens to predict the next one.
    * A **specialized Masked Cross-Entropy Loss** is the key. The model's error (loss) is **only calculated for the tokens it is supposed to generate (the assistant's response)**.

Let's visualize the loss calculation for the first turn:

| Input Context Seen by Model           | Next Token to Predict | Is Loss Calculated?                                        |
| ------------------------------------- | --------------------- | ---------------------------------------------------------- |
| `<s>[INST] What’s`                     | `the`                 | **NO** (This is the user's turn)                          |
| `...capital of Germany?`              | `[/INST]`             | **NO** (This is the user's turn)                           |
| `...Germany? [/INST]`                 | `The`                 | **YES!** (This is the start of the assistant's turn)       |
| `...Germany? [/INST] The`             | `capital`             | **YES!** (This is the assistant's turn)                    |
| `...is Berlin.`                       | `</s>`                | **YES!** (This is the end of the assistant's turn)         |

**Why this works:** You are not teaching the model how to *ask questions*; you are teaching it exclusively how to *answer* them, given the context of a user's question.
#

### Part 2: The Generation Phase (Inference / Having a Conversation)

**Goal:** Now that the model is fine-tuned, we can have a live, multi-turn conversation with it.

#### **Step 2.1: Start the Conversation (User's First Turn)**

We take the user's input and format it precisely as the model was trained, including the special tokens.

* **User Input:** `"What's the best way to get from San Jose to San Francisco?"`
* **Formatted for Model:** `<s>[INST] What's the best way to get from San Jose to San Francisco? [/INST]`

#### **Step 2.2: The Prediction Step (Model's First Response)**

The model processes the formatted input to generate its response one token at a time.

1.  **Linear Layer (Projecting to Vocabulary):** The model's final output is a vector of logits, with a size equal to its **vocabulary size** (e.g., 50,000). This represents a "score" for every possible next token.

2.  **Softmax Function:** The **softmax** function converts these scores into a probability distribution (`[0.01, 0.003, ..., 0.89, ...]`), showing the likelihood of each token being the correct next one.

3.  **Sampling Strategy:** We use a sampling method like **Top-p (Nucleus) Sampling** to choose a token from the probability distribution. This allows for fluent and natural-sounding responses.

#### **Step 2.3: The Conversational Loop (Maintaining Context)**

This is the most important part of a chatbot. The entire history is used to generate the next response.

1.  **Generate Full Response:** The model generates tokens autoregressively (`predict -> sample -> append`) until it produces an end-of-sequence token (`</s>`).
    * **Model's Response:** `"The best way depends on your priorities. Caltrain is a great option..."`

2.  **User's Next Turn:** The user replies.
    * **User's New Input:** `"I want to optimize for speed."`

3.  **Construct New Input:** **Crucially, you append the model's last answer and the user's new question to the history.** The new input fed to the model is the entire conversation so far, correctly formatted.
    `<s>[INST] What's the best way...? [/INST] The best way depends on...</s><s>[INST] I want to optimize for speed. [/INST]`

4.  **Repeat:** The model now generates a new response. Because it sees the *entire history*, it knows the user has prioritized speed and can give a tailored answer like, "In that case, driving outside of peak traffic hours is typically the fastest option." This loop continues, allowing the model to maintain context across many turns.

***

## ![GenAI](../../badges/text_classification.svg)

Use an LLM to perform text classification—the task of assigning a predefined category (like `Positive`, `Negative`, `Spam`, or `Legal`) to a piece of text. We will cover the two primary architectural approaches.
#

### Approach 1: The Encoder-Only Model (e.g., BERT)

This is the traditional and highly efficient method, optimized specifically for classification and understanding tasks.

#### **Part 1: The Training Phase** ⚙️

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

#### **Part 2: The Inference Phase** 💡

1.  **Format the Input:** A new, unseen piece of text is formatted with `[CLS]` and `[SEP]`.
    `[CLS] The flight was delayed again, and the staff was unhelpful. [SEP]`

2.  **Prediction:** The model performs a forward pass. It extracts the final embedding of the `[CLS]` token and passes it to the trained classification head. A **softmax** function is applied to the head's output to get a probability for each class.
    * **Output:** `[0.01, 0.99]` (Probabilities for Positive, Negative)

3.  **Final Label:** The label with the highest probability is chosen as the result.
    * **Result:** `Negative`
#

### Approach 2: The Decoder-Only Model (e.g., GPT, Llama)

This modern approach cleverly reframes classification as a text generation task.

#### **Part 1: The Training Phase** ✍️

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

#### **Part 2: The Inference Phase** 🔍

1.  **Format the Input:** The new text is placed into the prompt template, leaving the answer blank for the model to fill in.
    `Review: The flight was delayed again, and the staff was unhelpful. Sentiment:`

2.  **Prediction:** The model runs a forward pass and generates probabilities for the very next token. We are specifically interested in the probabilities of our potential label tokens (`Positive`, `Negative`, `Neutral`).
    * The model will output a probability distribution over the entire vocabulary.
    * We check the probabilities for our specific target words: `P("Positive") = 0.01`, `P("Negative") = 0.99`, `P("Neutral") = 0.00`.

3.  **Final Label:** We select the valid label token with the highest probability. This method ensures the model gives a constrained and valid answer.
    * **Result:** `Negative`

***

## ![GenAI](../../badges/nli.svg)
Natural Language Inference (NLI), breaking down the concepts, architecture, and processes in greater detail.
#

Natural Language Inference is a fundamental reasoning task in AI. Think of the model as a "logic detective" tasked with examining two statements: a piece of evidence (**the Premise**) and a claim (**the Hypothesis**). The model's job isn't to determine if the hypothesis is true in the real world, but only if it's true, false, or unrelated *based on the evidence provided in the premise*.

There are three possible verdicts:
* **`Entailment`**: The evidence proves the claim. The hypothesis logically follows from the premise.
* **`Contradiction`**: The evidence refutes the claim. The hypothesis logically contradicts the premise.
* **`Neutral`**: The evidence is irrelevant to the claim. The premise neither proves nor refutes the hypothesis.

#### **Example Data**

* **Premise:** `"A man in a blue shirt is riding a horse through a field."`
* **Hypothesis:** `"A person is outdoors."`
* **Target:** `Entailment` (Riding a horse through a field implies being outdoors).

#### **Use Case Scenarios**

NLI is a critical component for building more reliable and logical AI systems.
* **Validating AI-Generated Content:** If you ask an AI to summarize a legal document, you can use an NLI model to check if the summary's claims (`hypothesis`) are logically supported by the original text (`premise`). This helps prevent factual errors or "hallucinations."
* **Improving Search Engines:** NLI can help a search engine understand if a web page (`premise`) truly answers a user's query (`hypothesis`), going beyond simple keyword matching.
* **Fact-Checking Systems:** In a news article, if the premise is `"The company's profits soared to a record high in Q4"`, NLI can flag a user comment saying `"The company lost money in Q4"` as a `Contradiction`.

#

### Deep Dive: The Encoder-Only Architecture (The Gold Standard for NLI)

While other architectures can perform NLI, Encoder-Only models like BERT, RoBERTa, and DeBERTa are purpose-built for this kind of task and consistently achieve the best performance.

#### **Why Encoders Dominate NLI**

The key is **bi-directional context**.

Imagine you're solving a logic puzzle with two pieces of text. You wouldn't just read the first one and then the second one. You'd read both, then jump back and forth, comparing specific words and phrases. This is what an encoder does.

* **Bi-directional Attention:** Every token in the input can "see" and "attend to" every other token, regardless of position. This allows the model to create direct connections, for example, between the word `"horse"` in the premise and the word `"animal"` in the hypothesis.
* **Holistic Understanding:** An encoder doesn't process text left-to-right. It builds a holistic representation of the *entire input sequence at once*. For a comparison task like NLI, this is a massive advantage over decoder models that are optimized for sequential, next-token prediction.

#

### The Training Phase in Detail 

**Goal:** To fine-tune a pre-trained encoder model to specialize in comparing sentence pairs and predicting their logical relationship.

#### **Step 1: Data Preparation & Tokenization**

The input data is meticulously structured to be fed into the model.

1.  **Start with the Pair:**
    * `Premise:` "The event is scheduled for Tuesday."
    * `Hypothesis:` "The event is on the weekend."
2.  **Add Special Tokens:** The sentences are concatenated into a single string with special tokens that act as structural signposts.
    * `[CLS]` (Classification): A token added to the very beginning. Its purpose is to act as a "gathering point" for the entire sequence's meaning, specifically for classification tasks.
    * `[SEP]` (Separator): A token used to mark the boundary between the two sentences.
3.  **Final Formatted Input:**
    `[CLS] The event is scheduled for Tuesday. [SEP] The event is on the weekend. [SEP]`
4.  **Tokenization:** This string is converted into a sequence of numerical IDs by the model's tokenizer.

#### **Step 2: The Forward Pass Through the Encoder**

1.  **Input Embeddings:** The numerical IDs are converted into vectors. Positional embeddings are added to give the model a sense of word order.
2.  **Processing through Encoder Layers:** The sequence of vectors is passed through a stack of Transformer Encoder layers. In each layer, the embedding for every token is updated based on its relationship with all other tokens via self-attention.
3.  **The `[CLS]` Token's Journey:** The `[CLS]` token is treated like any other token initially. However, because it can attend to every word in both the premise and the hypothesis, by the time it exits the final layer, its output vector (e.g., a 768-dimensional vector) has become a rich, aggregated representation of the *relationship between the two sentences*.

#### **Step 3: The Classification Head & Loss Function**

1.  **The "Head":** We take the single output vector of the `[CLS]` token from the final encoder layer. This vector is fed into a "classification head," which is typically a very simple neural network (often just one linear layer) that is added on top of the pre-trained model.
2.  **Output Logits:** The head's job is to project the 768-dimensional `[CLS]` vector down to a vector with a size equal to the number of labels. For NLI, this is a vector of size 3. These raw output scores are called **logits**.
    * `[CLS] vector (size 768) -> Classification Head -> Logits (size 3)`
3.  **Loss Calculation:** To train the model, we use **Cross-Entropy Loss**. This function compares the logits (after a softmax is applied) to the correct "one-hot" label. If the correct label is `Contradiction` (e.g., index 1), the target is `[0, 1, 0]`. The loss function calculates a penalty based on how far the model's prediction is from this target. This penalty is then used to update the weights of the classification head and, typically, to fine-tune the weights of the entire encoder model as well.

#

### The Inference Phase in Detail 💡

**Goal:** To use our fine-tuned NLI model to predict the relationship for a new, unseen pair of sentences.

1.  **Format the Input:** A new premise-hypothesis pair is formatted in the exact same `[CLS]...[SEP]...[SEP]` structure.
2.  **Perform Forward Pass:** The formatted input is fed through the fine-tuned model.
3.  **Extract the `[CLS]` Vector:** We take the final output vector corresponding to the `[CLS]` token.
4.  **Predict with the Head:** This vector is passed through the trained classification head, producing a logit vector of size 3.
5.  **Apply Softmax for Probabilities:** The softmax function is applied to the logits to convert them into a human-readable probability distribution.
    * `Logits: [-4.6, 6.2, -3.1]` -> `Softmax` -> `Probabilities: [0.001, 0.998, 0.001]`
6.  **Interpret the Output:** The final prediction is the label corresponding to the highest probability.
    * **Result:** `{ "label": "Contradiction", "score": 0.998 }`

#
### Alternative Architectures: The Decoder-Only Approach

While encoders are the specialized tool, a powerful decoder-only model like GPT-4 can also perform NLI by framing it as a question-answering or completion task.

* **Prompt Engineering:** You would need to create a detailed prompt that instructs the model on what to do.
    ```
    Based on the Premise, determine if the Hypothesis is an entailment, contradiction, or neutral.

    Premise: A man in a blue shirt is riding a horse through a field.
    Hypothesis: A person is outdoors.

    The logical relationship is:
    ```
* **Inference:** The model would then generate the next token. A well-instructed model would produce the word "Entailment".

**Pros and Cons for NLI:**
* **Pros:** Extremely flexible, requires no special architecture, can use few-shot prompting to learn the task on the fly.
* **Cons:** The left-to-right nature is fundamentally less suited for direct comparison than an encoder's bi-directional view. It can be less efficient and potentially less robust than a fine-tuned encoder specifically optimized for the NLI task.

***
## ![GenAI](../../badges/qa.svg)

### 5a. Extractive Question Answering

#### **Example Data**
The data consists of a `Context` and a `Question`. The target is not a string of text, but rather the starting and ending positions (indices) of the answer within the context.

* **Context:** `"The first iPhone was introduced by Apple CEO Steve Jobs on January 9, 2007, at the Macworld convention."`
* **Question:** `"Who introduced the first iPhone?"`
* **Target:** `start_token_index: 10`, `end_token_index: 11` (Assuming "Steve" is the 10th token and "Jobs" is the 11th).

#### **Use Case Scenario**
The goal is to find and highlight a precise answer that already exists within a provided document. This is ideal for systems that need to be factual and grounded in a specific text.

* **A legal assistant is reviewing a contract:** The user asks, `"What is the termination clause notice period?"` The system scans the contract (`Context`) and extracts the exact phrase, like `"thirty (30) business days"`, highlighting it for the user.
* **Corporate knowledge base:** An employee asks the HR portal, `"What is the maximum rollover for vacation days?"` The system finds the answer directly in the employee handbook.
#

#### **How It Works: A Mini-Tutorial (Encoder-Only)**
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
#


### 1. How do we ensure `end_token_index >= start_token_index`?

You are absolutely right to ask this. If you simply take the `argmax` (the index with the highest score) of the start probabilities and the `argmax` of the end probabilities independently, you could easily end up with a `start_index` of 25 and an `end_index` of 10, which is nonsensical.

This is handled during the **inference/post-processing stage**. Here are the common strategies, from simple to robust:

#### Simple Approach: Check After Predicting
1.  Find the index of the highest start probability, let's call it `best_start_idx`.
2.  Find the index of the highest end probability, let's call it `best_end_idx`.
3.  **Check the condition:** If `best_end_idx < best_start_idx`, you conclude that the model could not find a valid answer and return an empty string or a "no answer found" message.

* **Drawback:** This is a bit naive. The model might have predicted a slightly lower-scoring but *valid* span (e.g., the second-best start and second-best end form a valid pair) that this approach would miss.

#### Robust Approach: Find the Best *Valid* Span (Most Common)

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

### 2. Do we take the softmax over the entire context length?

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
#
### 5b. Generative (Abstractive) Question Answering

#### **Example Data**
The data is typically a `Question` and a free-form `Answer`. The answer is not necessarily a direct quote from any text.

* **Input (Question):** `"Why is the sky blue?"`
* **Target (Answer):** `"The sky appears blue because of a phenomenon called Rayleigh scattering, where shorter-wavelength light, like blue and violet, is scattered more effectively by the tiny molecules of air in Earth's atmosphere."`

#### **Use Case Scenario**
The goal is to generate a natural, human-like answer, either by synthesizing information from a provided document (abstractive summarization) or by drawing from its own vast internal knowledge learned during pre-training. This is the foundation of modern conversational AI.

* **General knowledge queries:** A student uses Google or ChatGPT to ask, `"Explain the process of photosynthesis in simple terms."`
* **Customer support chatbot:** A user asks, `"How do I reset my account password?"` The chatbot generates a step-by-step guide instead of just pointing to a paragraph in a manual.

#
#### **How It Works: A Mini-Tutorial (Decoder-Only or Encoder-Decoder)**
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

## ![GenAI](../../badges/ie.svg)

#### **Example Data**

The goal of NER is to assign a categorical label to each token (or word) in a sentence. The data format consists of a sentence and a corresponding sequence of labels.

A common labeling format is the **IOB2 scheme**:
* **`B-TYPE`**: The **B**eginning of an entity of a certain TYPE (e.g., `B-PER` for Person).
* **`I-TYPE`**: **I**nside an entity of a certain TYPE. Used for entities that span multiple tokens.
* **`O`**: **O**utside of any entity.

* **Input Text:** `"Barack Obama was born in Hawaii."`
* **Tokenized Input:** `["Barack", "Obama", "was", "born", "in", "Hawaii", "."]`
* **Target Labels:** `[B-PER, I-PER, O, O, O, B-LOC, O]`

#### **Use Case Scenario**

The goal is to identify and pull out structured data (entities like people, organizations, dates, etc.) from unstructured, plain text.

* **An HR system processing a resume:**
    * **Input:** `"Worked as a software engineer at Google from 2019 to 2022."`
    * **System Extracts:** `{"Job Title": "software engineer", "Organization": "Google", "Start Date": "2019", "End Date": "2022"}`
* **News analysis:** Scanning thousands of articles to identify which companies are mentioned in relation to which locations.
* **Medical records:** Automatically extracting patient names, prescribed medications, and diagnoses from a doctor's notes.

#
#### **How It Works: A Mini-Tutorial (Encoder-Only)**
NER is a quintessential task for Encoder-Only models like BERT. The model needs to understand the full context of a word before it can classify it.

##### **Why Encoders are Ideal for NER**
The key is **bi-directional context**. To know if the word "Washington" refers to a person (`B-PER`) or a location (`B-LOC`), you need to see the words that come *after* it.
* "**Washington** spoke to Congress..." -> "Washington" is a person.
* "I am flying to **Washington** D.C...." -> "Washington" is a location.
An encoder model sees the entire sentence at once, allowing it to use both past and future context to make the most accurate decision for each word.

##### **The Training Phase ⚙️**

1.  **Input Formatting & Tokenization:** The input text is tokenized into a sequence of tokens (or more often, subwords). The target is a corresponding sequence of labels.
    * **Handling Subwords:** A critical detail is how to handle words that are broken into subwords (e.g., "Cupertino" -> `["Cuper", "##tino"]`). The standard approach is to assign the full label to the first subword (`B-LOC` for `"Cuper"`) and a special label (often `X` or simply the corresponding `I-` tag) to subsequent subwords (`I-LOC` for `"##tino"`). These subsequent subword predictions are often ignored during the loss calculation.
2.  **Architecture:** The model is a standard encoder. The unique part is the prediction head.
    * A single **classification head** is placed on top of **every token's final output vector**.
    * This head is a simple linear layer that projects the token's rich contextual embedding (e.g., a vector of size 768) to a vector of **logits**. The size of this logit vector is equal to the number of possible NER tags (e.g., 9 tags for `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `O`, etc.).
3.  **Loss Function:**
    * After the forward pass, you have a sequence of logit vectors—one for each input token.
    * A **Cross-Entropy Loss** is calculated **at each token position**. The model's predicted probability distribution for a given token is compared against the true label for that token.
    * The individual losses for all tokens in the sequence are then aggregated (usually by averaging) to get the final loss for the training step. The loss for padding tokens and subsequent subword tokens is ignored.

##### **The Inference Phase 💡**

1.  **Format and Predict:** A new sentence is tokenized and fed through the fine-tuned model. The model outputs a sequence of probability distributions, one for each input token.
2.  **Label Assignment:** For each token in the sequence, we perform an `argmax` on its probability distribution to find the most likely NER tag.
    * **Input:** `["Tim", "Cook", ",", "CEO", "of", "Apple"]`
    * **Raw Output Tags:** `[B-PER, I-PER, O, O, O, B-ORG]`
3.  **Post-Processing & Entity Extraction:** The raw sequence of tags is not the final output. A final, simple step is required to parse this sequence and group the tokens into structured entities.
    * The code iterates through the list of `(token, tag)` pairs.
    * When it sees a `B-PER` tag, it starts a new "Person" entity. It continues adding subsequent tokens with the `I-PER` tag to that same entity.
    * When it sees a different tag (like `O` or `B-ORG`), the current entity is considered complete.
    * This process converts the token-level predictions into the final, human-readable structured output.
    * **Final Structured Output:** `{"Person": "Tim Cook", "Organization": "Apple"}`

***

## ![GenAI](../../badges/sum.svg)

#### **Example Data**
The data consists of pairs of long documents and their corresponding shorter, human-written summaries.

* **Input (Article Text):**
    > "SAN JOSE – Following months of speculation, local tech giant Innovate Inc. today announced the release of their flagship product, the 'Quantum Leap' processor. The company claims the new processor is twice as fast as its predecessor and consumes 30% less power, a significant step forward for mobile computing. CEO Jane Doe presented the chip at a press event, stating that devices featuring the Quantum Leap will be available to consumers by the fourth quarter of this year."

* **Target (Summary):**
    > "San Jose-based Innovate Inc. has unveiled its 'Quantum Leap' processor, which is reportedly twice as fast and more power-efficient, with products expected to ship by Q4."

#### **Use Case Scenario**
The goal is to generate a short, coherent, and accurate summary from a longer document, capturing the most important information.

* **Executive Briefings:** A financial analyst feeds a 30-page quarterly earnings report into the system. The LLM outputs a one-paragraph executive summary highlighting key metrics like revenue, profit, and future outlook for a quick review by executives.
* **News Aggregation:** Summarizing multiple news articles about the same event to provide a comprehensive overview.
* **Meeting Productivity:** Condensing a one-hour meeting transcript into a short list of key decisions and action items.
* **Scientific Research:** Generating an abstract for a long scientific paper to help researchers quickly grasp its findings.

#
#### **How It Works:**
Text summarization is a classic sequence-to-sequence (seq2seq) task. It can be tackled effectively by two main architectural approaches.

### Approach 1: Encoder-Decoder Models (The Classic Approach)
Models like BART, T5, and the original Transformer were purpose-built for tasks like summarization and translation.

##### **Why the Encoder-Decoder Architecture is a Natural Fit**
This architecture has a clear division of labor that mirrors the summarization task itself:
* **The Encoder's Job:** To read and "understand" the entire source document. It processes the text using **bi-directional attention**, creating a rich, contextualized representation of the input. Its goal is to compress the full meaning of the article into a set of hidden states.
* **The Decoder's Job:** To generate a new, shorter text (the summary). It generates the summary one word at a time, but with a special power: **cross-attention**.

##### **The Training Phase ⚙️**

1.  **Encoding the Input:** The full text of the long article is fed into the Encoder. The encoder processes it and produces a set of output vectors (hidden states), one for each input token.
2.  **Decoding the Output:** The Decoder is an autoregressive model that is fed the target summary (using a technique called "teacher forcing"). At each step of generating a token, the decoder does two things:
    * It uses **causal self-attention** to look at the summary words it has already generated.
    * It uses **cross-attention** to look back at the encoder's output vectors. This allows the decoder to "consult" the most relevant parts of the original article while writing the summary, ensuring the output is factually grounded.
3.  **Loss Function:** A standard **Cross-Entropy Loss** is calculated by comparing the decoder's predicted tokens to the actual tokens in the human-written target summary.

##### **The Inference Phase 💡**

1.  The new, long document is fed into the encoder exactly once to get its representations.
2.  The decoder begins with a start-of-sequence `[SOS]` token.
3.  It generates the summary autoregressively (`predict -> sample -> append`), using cross-attention at each step to stay focused on the source article's main points. Generation stops when it produces an end-of-sequence `[EOS]` token.

#

### Approach 2: Decoder-Only Models (The Modern Approach)
Large models like GPT and Llama can perform summarization by framing it as a standard text completion task.

##### **Why This Works**
Through massive scale and instruction fine-tuning, these models learn to recognize the task of summarization from a prompt. The model learns the pattern: "When given a long text and the instruction to summarize, produce a shorter version containing the key points."

##### **The Training Phase ✍️**

1.  **Input Formatting:** The article and summary are concatenated into a single sequence using a prompt template.
    ```
    Summarize the following article:

    <text of the long news article>

    Summary:
    <text of the short summary>
    ```
2.  **Architecture & Loss:** The setup is identical to dialogue generation.
    * The model is a standard **decoder-only** architecture using **causal masking**.
    * The loss function is **Masked Cross-Entropy Loss**. The error is calculated **only on the tokens of the target summary**. This teaches the model to specifically generate the summary portion when prompted.

##### **The Inference Phase 🗣️**

1.  A new article is placed into the prompt template, stopping right after `Summary:`.
2.  The model takes this entire text as its input prefix.
3.  It begins generating the text that should follow, effectively writing the summary one token at a time using the standard autoregressive loop (`predict -> sample -> append`).


#
### Handling Long Documents During Training
That is the million-dollar question in applying LLMs to real-world documents! You have hit upon the single most significant practical limitation of standard Transformer models: their **fixed and finite context window**.

A model with a 4,000-token context window cannot "see" a 20,000-token document all at once. So, how do we train it on such data and use it to summarize long texts?

We use specialized strategies for both training and inference.

During the training phase, we have the luxury of knowing both the long document (`D`) and the target human-written summary (`S`). The goal is to create smaller, focused `(input, target)` pairs that fit within the model's context window.

The naive approach would be to simply truncate the document to the context window size. This is a terrible idea for summarization because the most important information might be at the end of the document.

A much better approach involves creating focused training examples. The most effective method is a form of "oracle" chunking:

1.  **Break down the Target Summary:** Take the human-written summary `S` and split it into individual sentences: `s_1, s_2, s_3, ...`
2.  **Find the "Oracle" Context:** For each summary sentence `s_i`, use an algorithm (like sentence-embedding similarity or a lexical overlap metric like ROUGE) to find the most relevant sentences or paragraphs from the original long document `D`. This selection of text from `D` is the "oracle context" because it's the specific information needed to write that part of the summary.
3.  **Create Training Pairs:** You can now create multiple high-quality training examples from one document-summary pair.
    * **Input 1:** The oracle context for `s_1`.
    * **Target 1:** The summary sentence `s_1`.
    * **Input 2:** The oracle context for `s_2`.
    * **Target 2:** The summary sentence `s_2`.
    * ...and so on.

This method ensures that every training example is a high-quality, focused task where the input contains the necessary information to generate the target, and everything fits within the context window.

#
### Handling Long Documents During Inference

During inference, we only have the long document and need to generate a single, coherent summary. Since the model can't see the whole document, we use chunking-based strategies.

#### Strategy 1: Map-Reduce (Chunk, Summarize, Summarize Again)

This is the most common and intuitive approach. It's a two-step process:

1.  **MAP Step:**
    * Split the long document into smaller, manageable chunks that fit the context window. It's best to use overlapping chunks to ensure you don't cut off sentences in the middle.
    * Send each chunk to the LLM *independently* with the prompt: `"Summarize the following text: <chunk_text>"`.
    * You will end up with a list of partial summaries, one for each chunk.

2.  **REDUCE Step:**
    * Combine all the partial summaries into a single new document.
    * Send this *combined summary document* to the LLM again with the same prompt: `"Summarize the following text: <combined_summaries_text>"`.
    * The final output is the summary of the summaries.

* **Analogy:** Several people each summarize one chapter of a book. Then, a final editor takes their chapter summaries and writes a single summary for the entire book.
* **Pros:** Simple to implement, highly parallelizable (you can summarize all chunks at the same time).
* **Cons:** Can lose important context that spans across two chunks. The quality of the final summary depends heavily on the quality of the intermediate summaries.

#### Strategy 2: Iterative Refinement

This method attempts to maintain a running context as it moves through the document.

1.  Summarize the first chunk (`chunk_1`). Let's call the result `summary_1`.
2.  Take `summary_1`, append the next chunk (`chunk_2`) to it, and feed it to the LLM with a prompt like: `"Given the following summary so far and the next part of the document, refine and update the summary: Summary: <summary_1> Next Part: <chunk_2>"`.
3.  The model produces a new, updated summary (`summary_2`).
4.  Repeat this process, feeding `summary_n` and `chunk_n+1` into the model until all chunks have been processed.

* **Pros:** Can capture cross-chunk relationships better than Map-Reduce.
* **Cons:** It's a sequential process and therefore much slower. It can also suffer from "recency bias," where information from later chunks is weighted more heavily.

### The Future: Native Long-Context Models

The strategies above are clever workarounds for a fundamental architectural limitation. The ultimate solution, which is rapidly becoming a reality, is to use models with much larger context windows.

Models like **Google's Gemini 1.5 Pro (with up to 1 million tokens)**, Anthropic's Claude series (100k-200k tokens), and specialized research models (Longformer, BigBird) are designed to handle much longer sequences.

For these models, you may not need a complex chunking strategy at all. You might be able to feed the entire document directly into the model's context window, allowing it to perform summarization with a full, holistic understanding of the text. This is the direction the field is moving.

***

## ![GenAI](../../badges/mt.svg)

#### **Example Data**
The data consists of parallel sentences: the same sentence in a source language and its human-written translation in a target language.

* **Input (Source Language - English):**
    > "The nearest Caltrain station is Diridon Station."

* **Target (Target Language - French):**
    > "La gare Caltrain la plus proche est la gare Diridon."

#### **Use Case Scenario**
The goal is to automatically translate text from one language to another, breaking down language barriers for communication, business, and information access.

* **Real-time Communication:** A tourist in San Jose uses Google Translate on their phone to ask for directions in English, and the app speaks the translated question in Spanish to a local resident.
* **Global Business:** A company translates its technical documentation and marketing materials from English into Japanese and German to reach international markets.
* **Information Access:** A researcher translates a scientific paper from Chinese into English to stay current with global developments in their field.

#
#### **How It Works: A Mini-Tutorial**
Machine Translation is the original, quintessential sequence-to-sequence (seq2seq) task that inspired the creation of the Transformer architecture.

### Approach 1: Encoder-Decoder Models (The Classic & Gold Standard)
Models like the original Transformer, MarianMT, and T5 are purpose-built for translation and are considered the gold standard for this task.

##### **Why the Encoder-Decoder Architecture is a Natural Fit**
This architecture's "read then write" division of labor is a perfect match for translation:
* **The Encoder's Job:** To read the entire source sentence (e.g., in English) using **bi-directional attention**. Its goal is to create a rich, language-agnostic representation of the sentence's meaning, compressing it into a set of contextual vectors.
* **The Decoder's Job:** To generate a new sentence in the target language (e.g., French). It does this autoregressively, using **cross-attention** to constantly refer back to the encoder's representation of the source sentence. This ensures the translation is faithful to the original meaning.

##### **The Training Phase ⚙️**

1.  **Encoding:** The source sentence (English) is tokenized and fed into the Encoder.
2.  **Decoding:** The target sentence (French) is fed to the Decoder for teacher forcing. At each step, the decoder uses **cross-attention** to consult the encoder's output, allowing it to align words and concepts (e.g., aligning "station" with "gare").
3.  **Loss Function:** A standard **Cross-Entropy Loss** is calculated between the decoder's predicted French tokens and the actual French tokens from the target sentence.

##### **The Inference Phase 💡**
The English sentence is encoded once. The decoder then starts with a `[START]` token and generates the French translation token by token until it produces an `[END]` token.

#
### Approach 2: Decoder-Only Models (The Modern Generalist)
Large language models like GPT and Llama can perform translation by treating it as an instruction-following task.

##### **Why This Works**
Through extensive fine-tuning, these models learn the pattern of translation from prompts. They understand that when given text in one language and asked to translate it to another, their task is to generate the corresponding text in the target language.

##### **The Training Phase ✍️**

1.  **Input Formatting:** The source and target sentences are concatenated into a single sequence using a prompt template.
    ```
    Translate the following English text to French:

    The nearest Caltrain station is Diridon Station.

    French translation:
    La gare Caltrain la plus proche est la gare Diridon.
    ```
2.  **Handling Unequal Lengths (Your Specific Question):**
    This is the crucial part. The English input and French output have different token lengths. How does the model handle this during training? The answer lies in the **Masked Loss Function**.

    * **Step 1: A Single Sequence:** The model doesn't see two separate sentences; it sees one long, continuous sequence of tokens: `[Prompt Tokens] + [Target Tokens]`.
    * **Step 2: Causal Masking:** A standard causal mask is applied. When predicting any token, the model can only see the tokens that came before it in this long sequence.
    * **Step 3: Intelligent Loss Calculation:** We use a "loss mask" to tell the model which predictions to learn from.
        * For all the tokens that are part of the prompt and the source sentence (e.g., `"Translate...Diridon Station."`), their loss is ignored. We don't need to teach the model how to write the prompt.
        * The Cross-Entropy Loss is **only calculated for the tokens in the target sentence** (e.g., `"La gare...gare Diridon."`).

    **In short, the unequal length doesn't matter.** The model is simply performing next-token prediction on a long sequence, and the loss mask cleverly focuses the learning *only* on the part of the sequence we want it to generate (the translation).

##### **The Inference Phase 🗣️**

1.  The new English sentence is placed into the prompt template, stopping right after `"French translation:"`.
2.  This entire text is fed to the model as a prefix.
3.  The model generates the translation autoregressively, token by token, until it completes the sentence and outputs an end-of-sequence token.
   
---

## ![GenAI](../../badges/code.svg)

### Code Generation

#### **Example Data**
The data for fine-tuning a code generation model consists of pairs of natural language instructions (often in comments or docstrings) and their corresponding code implementations.

* **Input (Natural Language Prompt):**
    ```python
    # Write a Python function that takes a list of numbers 
    # and returns a new list with only the even numbers.
    ```

* **Target (Code Completion):**
    ```python
    def get_even_numbers(numbers):
        """
        Filters a list of numbers, returning only the even ones.
        """
        even_numbers = []
        for number in numbers:
            if number % 2 == 0:
                even_numbers.append(number)
        return even_numbers
    ```

#### **Use Case Scenario**
The goal is to automatically generate correct, efficient, and syntactically valid code from a natural language description or a partial code snippet. This significantly speeds up the software development process.

* **AI Pair Programming (e.g., GitHub Copilot):** A developer is working in their code editor (like VS Code). They type a comment: `// Create a function to fetch user data from the API endpoint '/api/users'`. The AI assistant instantly generates the complete function with the correct syntax for making an HTTP request.
* **Natural Language Data Analysis:** A data scientist in a Jupyter Notebook types: `"Plot the average house price by neighborhood from the 'san_jose_housing' dataframe."` The model generates the necessary Python code using libraries like `pandas` and `matplotlib` to perform the calculation and create the visualization.
* **Automated Unit Testing:** A developer writes a function, and the AI can automatically generate a suite of unit tests to verify its correctness.

#
#### **How It Works: A Mini-Tutorial**
The core insight behind code generation is that **code is just a highly structured form of text**. It has a strict grammar, syntax, and logical patterns. Therefore, LLMs, which are expert pattern recognizers, are exceptionally good at this task. The dominant architecture is the **Decoder-Only** model.

##### **The Training Phase ✍️**

1.  **The Data:** Code models are pre-trained on a massive corpus of text and code. The data comes from two main sources:
    * **Public Code Repositories:** Billions of lines of code from sources like GitHub are used. The model learns the syntax, structure, and common patterns of many programming languages (`code -> code` prediction).
    * **Paired Data:** To learn how to follow instructions, models are specifically trained on pairs of natural language and code. This data is mined from docstrings, code comments, programming tutorials, and Q&A sites like Stack Overflow (`natural language -> code` prediction).

2.  **Tokenization:** Code models often use a specialized **tokenizer**. Unlike a standard text tokenizer, a code tokenizer is optimized to handle programming constructs like indentation (which is critical in Python), brackets (`{}`, `[]`, `()`), operators (`++`, `->`, `:=`), and common variable names.

3.  **Input Formatting:** The training data is formatted into a single continuous sequence, just like other generative tasks. For an instruction-following pair, it would look like:
    `"<instruction_comment>" <separator> "<code_implementation>"`

4.  **Architecture & Loss:** The setup is identical to other text generation tasks.
    * The model is a standard **decoder-only** architecture (e.g., GPT, Llama, Codex).
    * It uses **Causal Masking**, meaning when predicting the next token, it can only see the code and comments that came before it.
    * The loss function is **Cross-Entropy Loss**, calculated on the model's predictions for the next token against the actual next token in the training data. For instruction-following pairs, the loss might be calculated only on the code tokens (the completion), not the instruction tokens (the prompt).

##### **The Inference Phase (Writing Code) 👨‍💻**

1.  **The Prompt:** The user provides a prompt. This can be a natural language comment, a function signature, or the beginning of a line of code.
    * Example Prompt: `def send_email(recipient, subject, body):`

2.  **The Generation Loop:** The model takes this prompt as its initial input and begins to generate the code autoregressively, one token at a time.

3.  **The Autoregressive Process:** This is the same loop used in all generative tasks:
    * **Predict:** The model uses its final **linear layer** and a **softmax** function to get a probability distribution over all possible next tokens in its vocabulary.
    * **Sample:** A token is chosen from this distribution. For code generation, the sampling is often less random (using a lower "temperature") than for creative writing, because correctness and predictability are more important than creativity.
    * **Append:** The newly chosen token is appended to the sequence, and this new, longer sequence becomes the input for the next step.
    * The loop might generate: `"""Sends an email...` then `"""` then `import smtplib` and so on.

4.  **Stopping Condition:** The generation continues until the model determines the code block is logically complete (e.g., it has closed all brackets and returned from the function) or it generates a special end-of-sequence token.

---

## ![GenAI](../../badges/reason.svg)

#### **Example Data**
The key to teaching reasoning is the data format. Instead of just a question and a final answer, the target data includes the intermediate thinking steps. This is known as Chain of Thought (CoT).

* **Input (Question):**
    > `"If John has 5 apples and gives 2 to Mary, how many does he have left?"`

* **Target (Chain of Thought + Answer):**
    > `"[REASONING] John starts with 5 apples. He gives away 2 apples. To find out how many are left, we need to subtract the number of apples given away from the starting amount. The calculation is 5 - 2. [REASONING] 5 - 2 = 3. [ANSWER] The final answer is 3."`

#### **Use Case Scenario**
The goal is to solve problems that cannot be answered with a simple fact, but require logical, arithmetic, or commonsense steps.

* **Multi-step Math Problems:** A user prompts the model with a classic word problem from a local perspective:
    > `"A Caltrain leaves the San Jose Diridon station at 10:00 AM traveling north at 60 mph. A car leaves the same station at 11:00 AM, following the same route at 70 mph. At what time will the car catch up to the train?"`
* **The LLM provides a step-by-step breakdown:** It first calculates the train's one-hour head start (60 miles). Then it finds the relative speed of the car (10 mph). Finally, it divides the distance by the relative speed to find the time taken (6 hours) and calculates the final time (5:00 PM).
* **Other Uses:** Solving logic puzzles, debugging code by reasoning about the error, and planning complex strategies.

#
#### **How It Works:**
Reasoning is not a feature that is explicitly programmed into LLMs. It is an **emergent capability** of very large models, significantly enhanced by a technique called Chain of Thought (CoT) fine-tuning.

##### **What is an "Emergent Capability"?**
An emergent capability is a behavior that appears in large models that was not present in smaller models. It arises spontaneously from the sheer scale of the model's parameters and training data. The simple task of "predicting the next token," when performed over trillions of words, leads to the model learning complex underlying patterns of logic and reasoning.

##### **The Key Technique: Chain of Thought (CoT)**
The core idea behind CoT is simple but powerful: **forcing the model to "show its work."**

When a model tries to jump directly to a final answer for a complex problem, it's more likely to make a mistake. By fine-tuning the model to first generate the step-by-step reasoning and *then* the final answer, we teach it a more robust and reliable problem-solving process.

##### **Architecture**
Reasoning is an advanced sequential generation task, making it the domain of large **Decoder-Only models** like Google's Gemini, GPT-4, and Llama.

##### **The Training Phase (CoT Fine-Tuning) ✍️**

1.  **The Data:** The training data consists of pairs of `(Question, Full_CoT_Answer)`. This data is often meticulously created by humans to teach the model high-quality reasoning patterns.
2.  **Input Formatting:** The question and the full target (reasoning + answer) are formatted into a single continuous sequence.
3.  **Architecture & Loss Function:**
    * The model is a standard **decoder-only** architecture using **Causal Masking**.
    * The loss function is **Cross-Entropy Loss**, and this is the crucial part: the loss is calculated over the **ENTIRE target sequence**.
    * This means the model is penalized not just for getting the final answer `3` wrong, but for getting any token in the reasoning steps (e.g., `"subtract"`, `"-"`, `"="`) wrong. This forces the model to learn the *process* of logical deduction, not just to memorize final answers.

##### **The Inference Phase (Solving a New Problem) 🧠**

1.  **The Prompt:** The user provides only the question (e.g., the train problem).
2.  **The Generation Loop:** The model takes the question as its initial prompt and begins to generate the solution autoregressively.
3.  **The Autoregressive Process:**
    * Because the model has been trained to output reasoning first, the highest probability next tokens will naturally form the step-by-step thinking process. It doesn't jump to the answer because that's not the pattern it learned.
    * It generates token by token (`predict -> sample -> append`), laying out its entire chain of thought before concluding with the final answer.
    * For reasoning tasks, the sampling "temperature" is often set very low (close to 0) to make the output more deterministic and logically consistent.

##### **A Note on Few-Shot CoT Prompting**
For the most powerful models, you may not even need to fine-tune them. You can elicit reasoning behavior directly in the prompt by providing an example. This is called **few-shot prompting**.

**Example:**
```
Q: If John has 5 apples and gives 2 to Mary, how many does he have left?
A: John starts with 5 apples. He gives away 2. 5 - 2 = 3. The final answer is 3.

Q: If a train leaves San Jose at 10:00 AM traveling at 60 mph... at what time will the car catch up?
A: [The model will now generate the step-by-step reasoning because it follows the format of the example provided.]
```
---
Of course. Here is a comprehensive mini-tutorial on Sentence-Transformers that synthesizes our entire discussion, including the architecture, use cases, different training philosophies, and the mathematical details of the loss functions.

***

## ![GenAI](../../badges/embedding.svg)

A standard transformer model like BERT is excellent at understanding words in context (token-level embeddings), but it is not inherently designed to create a single, meaningful vector for an entire sentence that can be easily compared with others. Sentence-Transformers solve this problem by fine-tuning these models to produce high-quality, sentence-level embeddings.

#### **Example Data**
The training data format depends on the training objective.

* **For Supervised Training (on similarity):**
    * `Sentence A:` "The weather in San Jose is sunny today."
    * `Sentence B:` "It is bright and clear in San Jose right now."
    * `Target Label (y):` `1.0` (indicating high similarity)

* **For Self-Supervised Training (using triplets):**
    * `Anchor:` "What is the capital of California?"
    * `Positive (Similar):` "Sacramento is the capital of California."
    * `Negative (Dissimilar):` "The Golden Gate Bridge is in San Francisco."

#### **Use Case Scenario**
The goal is to convert a sentence into a fixed-size numerical vector (an embedding) where sentences with similar meanings have vectors that are close together in a high-dimensional space. This enables powerful applications:

* **Semantic Search:** A user on a support website asks, `"How do I reset my password if I lost my 2FA device?"` The system converts this query into an embedding and instantly finds the most semantically similar questions and answers in its knowledge base, providing the exact solution.
* **Duplicate Detection:** A platform like Stack Overflow can identify if a newly asked question is a semantic duplicate of an existing one, even if they use different words.
* **Clustering:** Grouping thousands of news articles, user reviews, or documents by the topic or event they describe. This is a core component of modern RAG (Retrieval-Augmented Generation) systems.

#
#### **How It Works: A Mini-Tutorial**

##### **Architecture**
The architecture is a two-stage process:

1.  **Transformer Base:** A sentence is fed into a pre-trained transformer model like BERT or RoBERTa. This model uses bi-directional attention to process the entire sentence and outputs a contextualized vector for every token.
2.  **Pooling Layer:** To get a single vector for the entire sentence, the output token vectors are "pooled." The most common method is **mean pooling**, where you simply take the element-wise average of all the token vectors to produce one fixed-size sentence embedding.

##### **The Training Phase**
This is the most critical part. The model is fine-tuned to ensure the final pooled embeddings are semantically meaningful. This can be done with or without human-provided labels.

**1. Supervised Fine-Tuning (Requires Labels)**
This is the most common approach for state-of-the-art performance, where the model learns to mimic human judgments.

* **Process:** A **Siamese Network** structure is used. Two sentences, `s_1` and `s_2`, are passed through the *exact same* Sentence-Transformer model (with shared weights) to produce two embeddings, `u` and `v`.
* **Loss Function: Cosine Similarity Loss.** The goal is to make the model's predicted similarity score match a human-provided score `y`.
    * **Normalization:** For cosine similarity to work correctly, the embedding vectors `u` and `v` must be normalized to have a length of 1. This can be done as a final layer in the model or as a step within the loss function.
    * **Mathematical Formulation:** The cosine similarity between two vectors is their dot product divided by the product of their magnitudes.
$$
\text{cos_sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}
$$
        The model is trained to minimize a regression loss, typically **Mean Squared Error (MSE)**, between its predicted similarity and the ground-truth label `y`.

$$
L_{\text{cosine}} = \frac{1}{N} \sum_{i=1}^{N} (\text{cos_sim}(u_i, v_i) - y_i)^2
$$

**2. Self-Supervised / Unsupervised Fine-Tuning (No `y` score needed)**
This approach uses the structure of the data itself to create learning signals. The Triplet Network is a classic example.

* **Process:** A **Triplet Network** structure is used. Three sentences—an `anchor` (a), a `positive` (p), and a `negative` (n)—are passed through the same model to get three embeddings: `a`, `p`, and `n`.
* **Loss Function: Triplet Loss.** The goal is to "push" the negative embedding away from the anchor and "pull" the positive embedding closer. The loss function ensures that the anchor-to-positive distance is smaller than the anchor-to-negative distance by at least a certain `margin` (α).
    * **Mathematical Formulation:** Using Euclidean distance $d(x, y) = ||x - y||_2$, the objective is: $d(a, p) + α < d(a, n)$. The loss function that enforces this is:
$$
L_{\text{triplet}} = \max(0, d(a, p)^2 - d(a, n)^2 + \alpha)
$$
    * The loss is zero if the condition is already met. Otherwise, the model is penalized, forcing it to adjust the embedding space correctly.

##### **The Inference Phase**
Once fine-tuned, using the model is simple and fast:
1.  **Input:** A new sentence.
2.  **Process:** Feed the sentence through the fine-tuned Sentence-Transformer.
3.  **Output:** A single, fixed-size vector embedding that captures the sentence's meaning, ready to be used for search, clustering, or other downstream tasks.
***

## ![GenAI](../../badges/rag.svg)

It's important to note that RAG is not a type of model training like the previous examples, but rather a **system architecture** that combines a retrieval system with a generative LLM to produce better answers.

The core idea behind RAG is to solve a major weakness of LLMs: they can only answer based on the data they were trained on, which might be outdated or not include your private information. RAG gives the LLM an "open book" to use when answering a question, ensuring the response is timely, accurate, and grounded in specific facts.

#### **Example Data & Components**
RAG doesn't use a simple `Input -> Target` training pair. Instead, it's a system composed of several parts.

1.  **The Knowledge Base (The "Book"):** A collection of your private or specific documents.
    * *Document 1:* "San Jose City Council Meeting Minutes - June 10, 2025.pdf" -> *Content: "...the council voted to approve the new zoning proposal for the downtown area, project 'Urban Sky,' with a 7-4 majority..."*
    * *Document 2:* "Urban Sky Project Details.docx" -> *Content: "The Urban Sky project includes provisions for 30% affordable housing units..."*
    * *Document 3:* "Local Business Impact Report - June 2025.pdf"

2.  **The User Query (The "Question"):** A question that can only be answered using the knowledge base.
    > "What was the city council's final decision on the 'Urban Sky' zoning proposal in San Jose, and did it include affordable housing?"

3.  **The Final LLM Output (The "Answer"):** A synthesized answer based *only* on the retrieved information.
    > "Based on the meeting minutes from June 10, 2025, the San Jose City Council approved the 'Urban Sky' zoning proposal with a 7-4 majority. The project details confirm that it includes a provision for 30% affordable housing units."

#### **Use Case Scenario**
The goal of RAG is to ground a powerful generative model in a specific, up-to-date, or private set of documents, making its answers more trustworthy and factual.

* **Enterprise Chatbot:** An employee at a local company like Adobe or Cisco asks an internal chatbot, `"What is our corporate travel reimbursement policy for international flights updated for 2025?"` The RAG system retrieves the latest HR document from the company's internal server, ignores any old policies from the LLM's training data, and provides the current, correct answer.
* **Up-to-Date Customer Support:** A customer asks a support bot on a website, `"Is the new 'Quantum Leap' processor compatible with my motherboard model?"` The RAG system pulls the latest product manuals and compatibility charts to provide an accurate technical answer, even if the product was released after the LLM was trained.
* **Personalized Tutoring:** A student uploads their lecture notes and textbook chapters. They can then ask questions like, `"Explain Professor Smith's view on economic policy from last week's lecture,"` and the system will answer based only on that provided material.

---
#### **How It Works:**
RAG is primarily an **inference-time** process that involves two main phases: preparing the knowledge base (Indexing) and answering the query (Retrieval & Generation).

##### **Phase 1: The Indexing Phase (Offline Preparation - "Building the Library")**
This is a one-time setup process you perform on your knowledge base.

1.  **Load Documents:** Your system ingests your source documents (PDFs, Word docs, website pages, etc.).
2.  **Chunk Documents:** The documents are broken down into smaller, more manageable chunks (e.g., paragraphs or sentences). This is crucial because a small, focused chunk is more likely to match a specific question than a whole document.
3.  **Embed Chunks:** Each chunk of text is passed through an **embedding model** (e.g., a Sentence-Transformer). This model converts the text into a numerical vector (an embedding) that captures its semantic meaning.
4.  **Store in a Vector Database:** These chunks and their corresponding embeddings are stored in a specialized **vector database** (like Pinecone, Weaviate, or ChromaDB). This database is highly optimized for finding vectors that are "close" to each other in meaning.

##### **Phase 2: The Retrieval & Generation Phase (Real-time Inference)**
This happens every time a user asks a question.

1.  **Receive User Query:** The system receives the user's question.
2.  **Embed the Query:** The user's question is passed through the *same* embedding model to convert it into a query vector.
3.  **Retrieve Relevant Chunks:** The system uses this query vector to search the vector database. The database performs a similarity search and returns the `top-k` (e.g., the top 3-5) most relevant chunks from the knowledge base. This is the **RETRIEVAL** step.
4.  **Augment the Prompt:** A new, detailed prompt is constructed for the generative LLM. This prompt combines the retrieved context with the original question.
    ```
    Prompt Template:

    "You are a helpful assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say you don't know.

    Context:
    <Retrieved Chunk 1 from the meeting minutes>
    <Retrieved Chunk 2 from the project details>

    Question:
    What was the city council's final decision on the 'Urban Sky' zoning proposal in San Jose, and did it include affordable housing?"
    ```
5.  **Generate the Answer:** This augmented prompt is sent to a powerful generative LLM (like Gemini). The LLM reads the context and synthesizes a final answer based *only* on that information. This is the **GENERATION** step.

By providing the relevant text directly in the prompt, RAG constrains the LLM, dramatically reducing hallucinations (making things up) and ensuring the answer is grounded in the provided facts.

#

## ![GenAI](../../badges/multimodal.svg)

* **The Goal:** To understand and generate content that involves more than one modality, typically text and images.
* **Data Format:** The data consists of pairs of images and corresponding text.
    * **Input:** `(<image of a golden retriever playing in a park>, "What kind of dog is this?")`
    * **Target:** `"This is a Golden Retriever."`
* **Example Use Case:**
    * **(User uploads a photo of their lunch)**
    * **Prompt:** `"Estimate the calories in this meal."`
    * **LLM Output:** `"This meal appears to be a grilled chicken salad with avocado and a light vinaigrette. I would estimate it to be around 450-550 calories."`
* **How It Works:**
    * **Architecture:** This requires a specialized **Vision-Language Model (VLM)**, such as GPT-4o or Gemini. These models have two main components:
        1.  A **Vision Encoder** (like ViT) that processes the input image and converts it into a sequence of numerical embeddings.
        2.  A standard **Language Model** (typically Decoder-Only) that takes both the text embeddings and the image embeddings as its input sequence, allowing it to reason about the combined information.
