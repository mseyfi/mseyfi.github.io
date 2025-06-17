## A Comprehensive Guide to LLM Tasks: From Fine-Tuning to Advanced Applications

A single, powerful pre-trained Large Language Model is a versatile foundation that can be adapted—or "fine-tuned"—to excel at a wide array of specific tasks. The key to this versatility lies in how we frame the task and format the data.

This tutorial provides a detailed breakdown of the most common and important tasks an LLM can perform, explaining for each: its goal, the data format for fine-tuning, a practical example, and the underlying mechanics.

---

### 1. Text Generation (and Completion)

* **The Goal:** To produce creative, coherent, and contextually relevant text that continues from a given prompt. This is the most fundamental autoregressive task and includes applications like story writing, paraphrasing, and email completion.
* **The Data Format (for fine-tuning):** The data consists of high-quality prompt-completion pairs.
    * **Input:** `"The discovery of gravity began when"`
    * **Target:** `"Newton observed an apple falling from a tree..."`
* **Example Use Case:**
    * **Prompt:** `"In a world where cats rule the internet, one kitten dreamed of..."`
    * **LLM Output:** `"...becoming a professional astronaut. Her name was Mittens, and she spent her days staring at the moon, ignoring the laser pointers and catnip mice that amused her peers."`
* **How It Works:**
    * **Architecture:** The quintessential task for **Decoder-Only** models (like the GPT series, Llama).
    * **Structure:** The model is fed the input and target as a single continuous sequence. A **causal mask** ensures that when predicting each token, the model can only see the tokens that came before it.
    * **Loss Function:** A **Masked Cross-Entropy Loss** is used, where the loss is only calculated on the tokens in the target completion, forcing the model to learn to generate the desired output given the prompt.

---

### 2. Dialogue Generation (Chatbots)

* **The Goal:** To create an interactive, multi-turn conversational agent that can remember context, answer questions, and maintain a consistent persona.
* **The Data Format:** The data is a sequence of conversational turns, with roles clearly delineated.
    * **Input:** `<s>[INST] What’s the capital of Germany? [/INST]`
    * **Target:** `The capital of Germany is Berlin.</s>`
* **Example Use Case:** This is the core of systems like ChatGPT, Claude, and Google's Gemini.
    * **User:** `"What's the best way to get from San Jose to San Francisco?"`
    * **LLM:** `"The best way depends on your priorities. Caltrain is a great option for avoiding traffic and relaxing, while driving gives you more flexibility. Are you optimizing for speed, cost, or convenience?"`
* **How It Works:**
    * **Architecture:** Dominated by **Decoder-Only** models.
    * **Structure:** This is a more structured form of text generation. **Special tokens** (e.g., `[INST]`, `[USER]`, `[ASSISTANT]`) are used to define the roles in the conversation. The entire history is fed to the model to maintain context.
    * **Loss Function:** The **Masked Cross-Entropy Loss** is crucial here. The loss is **only calculated on the assistant's turns**, teaching the model how to respond helpfully and correctly.

---

### 3. Text Classification

* **The Goal:** To assign a single, predefined categorical label to an entire piece of text. This includes sentiment analysis, topic classification, and spam detection.
* **The Data Format:** A pair of text and its corresponding label.
    * **Input:** `"I love this product! It works perfectly."`
    * **Target:** `Positive`
* **Example Use Case:**
    * **Input:** `"The flight was delayed again, and the staff was unhelpful."`
    * **LLM Output:** `{ "label": "Negative", "score": 0.99 }`
* **How It Works:**
    * **Architecture 1: Encoder-Only (e.g., BERT):** This is the classic, high-performance approach. The input is prepended with a `[CLS]` token. The final hidden state of this `[CLS]` token, which acts as an aggregate representation of the sequence, is fed into a simple classification head. The loss is **Cross-Entropy Loss** on the class label.
    * **Architecture 2: Decoder-Only (e.g., GPT):** The task is reframed as text completion (e.g., `Review: ... Sentiment: Negative`). The loss is calculated **only on the token(s) for the target label** ("Negative").

---

### 4. Natural Language Inference (NLI)

* **The Goal:** A specialized classification task to determine the logical relationship between two sentences: a *premise* and a *hypothesis*.
* **The Data Format:** A pair of sentences and a label (`Entailment`, `Contradiction`, or `Neutral`).
    * **Input:** `Premise: "A man in a blue shirt is riding a horse." Hypothesis: "A man is on an animal."`
    * **Target:** `Entailment`
* **Example Use Case:** Used for logical consistency checking in other AI systems.
    * **Input:** `Premise: "The event is scheduled for Tuesday." Hypothesis: "The event is on the weekend."`
    * **LLM Output:** `{ "label": "Contradiction" }`
* **How It Works:**
    * **Architecture: Encoder-Only:** This is the most effective approach. The two sentences are concatenated using a `[SEP]` token: `[CLS] premise [SEP] hypothesis [SEP]`. The `[CLS]` token's final hidden state is then used for classification, just like in standard text classification.
    * **Loss Function:** **Cross-Entropy Loss** on the three possible labels.

---

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

---

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

---

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

---

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

---

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

---

### 10. Reasoning Tasks

* **The Goal:** To solve problems that require logical, arithmetic, or commonsense steps.
* **Data Format (for Chain-of-Thought fine-tuning):** The data includes the intermediate reasoning steps.
    * **Input:** `"If John has 5 apples and gives 2 to Mary, how many does he have left?"`
    * **Target:** `"John starts with 5 apples. He gives away 2. So we need to subtract 2 from 5. 5 - 2 = 3. The answer is 3."`
* **Example Use Case:**
    * **Prompt:** `"If a train leaves San Jose at 10:00 AM traveling at 60 mph and a car leaves at 11:00 AM traveling at 70 mph on the same route, at what time will the car catch up to the train?"`
    * **LLM Output:** A step-by-step breakdown of the relative speed, the head start of the train, and the final calculation for the time taken.
* **How It Works:** This is an **emergent capability** of very large models. It is significantly improved by fine-tuning on datasets that explicitly include a **Chain of Thought (CoT)**. By training the model to generate the reasoning steps *before* the final answer, it learns a more robust process for solving complex problems. The loss is calculated on the entire generated sequence (reasoning + answer).

---

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

---

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
