[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# A Comprehensive Guide to LLM Tasks: From Fine-Tuning to Advanced Applications

A single, powerful pre-trained Large Language Model is a versatile foundation that can be adapted—or "fine-tuned"—to excel at a wide array of specific tasks. The key to this versatility lies in how we frame the task and format the data.

This tutorial provides a detailed breakdown of the most common and important tasks an LLM can perform, explaining for each: its goal, the data format for fine-tuning, a practical example, and the underlying mechanics.

***

## 1. Text Generation (and Completion)

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

## 2. Dialogue Generation (Chatbots)


### Mini-Tutorial: Training a Dialogue LLM (Chatbot)

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
| ############- | ####### | ###################- |
| `<s>[INST] What’s`                     | `the`                 | **NO** (This is the user's turn)                           |
| `...capital of Germany?`              | `[/INST]`             | **NO** (This is the user's turn)                           |
| `...Germany? [/INST]`                 | `The`                 | **YES!** (This is the start of the assistant's turn)       |
| `...Germany? [/INST] The`             | `capital`             | **YES!** (This is the assistant's turn)                      |
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

## 3. Text Classification
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

### 4. Natural Language Inference (NLI)
Of course. Here is an expanded, in-depth tutorial on Natural Language Inference (NLI), breaking down the concepts, architecture, and processes in greater detail.

#

### Expanded Tutorial: Natural Language Inference (NLI)

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

### The Training Phase in Detail ⚙️

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

1.  **The "Head":** We take the single output vector of the `[CLS]` token from the final encoder layer. This vector is fed into a "classification head," which is typically a very simple neural network (often just one linear layer and a non-linear activation function like Tanh or ReLU) that is added on top of the pre-trained model.
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
### 5. Question Answering (QA)

* **The Goal:** To provide a precise answer to a user's question, either based on internal knowledge or a provided context.

#### 5a. Extractive QA
* **Goal:** To *extract* the answer span directly from a given context document.
* **Data Format:** `(Context, Question) -> (start_token_index, end_token_index)`
* **Example Use Case:**
    * **Context:** `"The first iPhone was introduced by Apple CEO Steve Jobs on January 9, 2007, at the Macworld convention."`
    * **Question:** `"Who introduced the first iPhone?"`
    * **LLM Output:** `"Steve Jobs"` (extracted directly from the text).
* **How It Works:**
    * **Architecture:** Best suited for **Encoder-Only** models. The model processes the context and question together. Two separate classification heads are placed on top of every token's output embedding to predict the probability of that token being the `start` of the answer and the `end` of the answer.

#### 5b. Generative (Abstractive) QA
* **Goal:** To *generate* a free-form answer based on provided context or its own internal knowledge.
* **Data Format:** `(Question) -> (Answer)` or `(Context, Question) -> (Answer)`
* **Example Use Case:**
    * **Q:** `"Why is the sky blue?"`
    * **A:** `"The sky appears blue because of a phenomenon called Rayleigh scattering, where shorter-wavelength light, like blue and violet, is scattered more effectively by the tiny molecules of air in Earth's atmosphere."`
* **How It Works:**
    * **Architecture:** A classic task for **Decoder-Only** or **Encoder-Decoder** models. The question (and context, if provided) is formatted into a prompt, and the model is fine-tuned to generate the answer. The loss is **Masked Cross-Entropy** on the answer tokens.

#

### 6. Information Extraction

* **The Goal:** To identify and pull structured data, like names, dates, or relationships, from unstructured text.
* **Data Format (for Named Entity Recognition - NER):**
    * **Input:** `"Barack Obama was born in Hawaii."`
    * **Target:** A sequence of labels, one per token: `[B-PER, I-PER, O, O, O, B-LOC]`
* **Example Use Case (NER):**
    * **Input:** `"Tim Cook, the CEO of Apple, announced the new device from their headquarters in Cupertino."`
    * **LLM Output:** `Person: "Tim Cook"`, `Organization: "Apple"`, `Location: "Cupertino"`
* **How It Works:**
    * **Architecture: Encoder-Only (e.g., BERT):** This is the ideal architecture due to its bidirectional context. A classification head is placed on top of **every token's** output vector to predict its entity label (e.g., `B-PER`, `I-PER`, `O`).
    * **Loss Function:** **Cross-Entropy Loss** is calculated across the entire sequence of token labels.

#

### 7. Summarization

* **The Goal:** To generate a short, coherent, and accurate summary from a longer document.
* **Data Format:** Pairs of long documents and their human-written summaries.
    * **Input:** `"<text of a long news article>"`
    * **Target:** `"<text of the short summary>"`
* **Example Use Case:**
    * **Input:** A 30-page quarterly financial report.
    * **LLM Output:** A one-paragraph executive summary highlighting key metrics like revenue, profit, and future outlook.
* **How It Works:**
    * **Architecture:** A classic sequence-to-sequence task, perfectly suited for **Encoder-Decoder** models but also commonly handled by **Decoder-Only** models (see Section 2 for details).

#

### 8. Machine Translation

* **The Goal:** To translate a sequence of text from a source language to a target language.
* **Data Format:** Parallel sentences or documents.
    * **Input:** `"translate English to Spanish: The weather today is sunny and warm."`
    * **Target:** `"El clima hoy está soleado y cálido."`
* **Example Use Case:**
    * **Input:** `"I would like to book a hotel room for two nights."` (English to Japanese)
    * **LLM Output:** `"ホテルの部屋を二泊予約したいです。"`
* **How It Works:**
    * **Architecture:** The quintessential task for **Encoder-Decoder** models. The encoder gets a full understanding of the source sentence, and the decoder generates the target language.

#

### 9. Code-Related Tasks

* **The Goal:** To generate, complete, document, or translate code.
* **Data Format (for generation):** A natural language description paired with the corresponding code.
    * **Input:** `"Python function to calculate the nth Fibonacci number"`
    * **Target:** `"def fibonacci(n): ..."`
* **Example Use Case:**
    * **Prompt:** `"// A Javascript function that fetches data from a URL and logs it to the console"`
    * **LLM Output (Code Completion):**
        ```javascript
        async function fetchData(url) {
          try {
            const response = await fetch(url);
            const data = await response.json();
            console.log(data);
          } catch (error) {
            console.error('Error fetching data:', error);
          }
        }
        ```
* **How It Works:**
    * **Architecture:** This is a specialized application of **Decoder-Only** models. They are pre-trained on massive corpora of publicly available code (e.g., from GitHub) in addition to natural language text. The fine-tuning process is identical to text generation (Category 1).

#

### 10. Reasoning Tasks

* **The Goal:** To solve problems that require logical, arithmetic, or commonsense steps.
* **Data Format (for Chain-of-Thought fine-tuning):** The data includes the intermediate reasoning steps.
    * **Input:** `"If John has 5 apples and gives 2 to Mary, how many does he have left?"`
    * **Target:** `"John starts with 5 apples. He gives away 2. So we need to subtract 2 from 5. 5 - 2 = 3. The answer is 3."`
* **Example Use Case:**
    * **Prompt:** `"If a train leaves San Jose at 10:00 AM traveling at 60 mph and a car leaves at 11:00 AM traveling at 70 mph on the same route, at what time will the car catch up to the train?"`
    * **LLM Output:** A step-by-step breakdown of the relative speed, the head start of the train, and the final calculation for the time taken.
* **How It Works:** This is an **emergent capability** of very large models. It is significantly improved by fine-tuning on datasets that explicitly include a **Chain of Thought (CoT)**. By training the model to generate the reasoning steps *before* the final answer, it learns a more robust process for solving complex problems. The loss is calculated on the entire generated sequence (reasoning + answer).

#

### 11. Retrieval-Augmented Generation (RAG)

* **The Goal:** To ground the LLM's responses in external, verifiable knowledge, reducing hallucinations and allowing it to use up-to-date or private information.
* **Data Format:** This is not a fine-tuning task but an **inference-time architecture**.
* **Example Use Case:**
    * **User Question:** `"What were the key features announced for the iPhone 16 in today's keynote?"`
    * **RAG System Process:**
        1.  **Retrieve:** The system searches a database of recent news articles and finds several articles about the keynote from today, June 16, 2025.
        2.  **Augment:** It combines the text from these articles with the user's question into a new, large prompt.
        3.  **Generate:** The LLM receives the augmented prompt and generates a summary based *only* on the provided text.
    * **LLM Output:** `"According to the keynote reports, the key features announced for the iPhone 16 include a new A19 Bionic chip, a periscope zoom lens for the standard model, and an always-on display with dynamic widgets."`
* **How It Works:** RAG is a **system-level pattern** that combines a **retriever** (e.g., a search engine) with a standard **generative LLM**. The LLM's role is essentially to perform high-quality reading comprehension on the retrieved documents.

#

### 12. Multimodal Tasks

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
