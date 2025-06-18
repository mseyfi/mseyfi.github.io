[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Dialogue Generation (Chatbot)

This tutorial explains how to take a general-purpose LLM and fine-tune it to be an interactive, multi-turn conversational agent that can remember context and adopt a specific persona.

---

#### Part 1: The Training Phase (Fine-Tuning for Dialogue)

**Goal:** To teach a pre-trained model the structure, flow, and turn-taking nature of a human conversation. The model learns to act as a helpful assistant.

##### **Step 1.1: Prepare the Data (The Conversational Script)**

You start with a dataset of multi-turn dialogues. These dialogues are formatted with **special tokens** to clearly define who is speaking. This structure is critical for the model to learn its role.

Let's look at the format from your example (`<s>` is start of sequence, `</s>` is end of sequence, `[INST]` marks a user instruction).

**Example Dialogue:**
```
<s>[INST] What’s the capital of Germany? [/INST] The capital of Germany is Berlin.</s><s>[INST] What about France? [/INST] The capital of France is Paris.</s>
```

* **Special Tokens:** `<s>`, `[INST]`, `[/INST]`, `</s>` are not words; they are structural signposts. They teach the model: "When you see text between `[INST]` and `[/INST]`, that's the user speaking. Your job is to provide the text that comes after `[/INST]`."

##### **Step 1.2: The Training Process**

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

---

#### Part 2: The Generation Phase (Inference / Having a Conversation)

**Goal:** Now that the model is fine-tuned, we can have a live, multi-turn conversation with it.

##### **Step 2.1: Start the Conversation (User's First Turn)**

We take the user's input and format it precisely as the model was trained, including the special tokens.

* **User Input:** `"What's the best way to get from San Jose to San Francisco?"`
* **Formatted for Model:** `<s>[INST] What's the best way to get from San Jose to San Francisco? [/INST]`

##### **Step 2.2: The Prediction Step (Model's First Response)**

The model processes the formatted input to generate its response one token at a time.

1.  **Linear Layer (Projecting to Vocabulary):** The model's final output is a vector of logits, with a size equal to its **vocabulary size** (e.g., 50,000). This represents a "score" for every possible next token.

2.  **Softmax Function:** The **softmax** function converts these scores into a probability distribution (`[0.01, 0.003, ..., 0.89, ...]`), showing the likelihood of each token being the correct next one.

3.  **Sampling Strategy:** We use a sampling method like **Top-p (Nucleus) Sampling** to choose a token from the probability distribution. This allows for fluent and natural-sounding responses.

##### **Step 2.3: The Conversational Loop (Maintaining Context)**

This is the most important part of a chatbot. The entire history is used to generate the next response.

1.  **Generate Full Response:** The model generates tokens autoregressively (`predict -> sample -> append`) until it produces an end-of-sequence token (`</s>`).
    * **Model's Response:** `"The best way depends on your priorities. Caltrain is a great option..."`

2.  **User's Next Turn:** The user replies.
    * **User's New Input:** `"I want to optimize for speed."`

3.  **Construct New Input:** **Crucially, you append the model's last answer and the user's new question to the history.** The new input fed to the model is the entire conversation so far, correctly formatted.
    `<s>[INST] What's the best way...? [/INST] The best way depends on...</s><s>[INST] I want to optimize for speed. [/INST]`

4.  **Repeat:** The model now generates a new response. Because it sees the *entire history*, it knows the user has prioritized speed and can give a tailored answer like, "In that case, driving outside of peak traffic hours is typically the fastest option." This loop continues, allowing the model to maintain context across many turns.

***

