[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Text Summarization

##### **Example Data**
The data consists of pairs of long documents and their corresponding shorter, human-written summaries.

* **Input (Article Text):**
    > "SAN JOSE – Following months of speculation, local tech giant Innovate Inc. today announced the release of their flagship product, the 'Quantum Leap' processor. The company claims the new processor is twice as fast as its predecessor and consumes 30% less power, a significant step forward for mobile computing. CEO Jane Doe presented the chip at a press event, stating that devices featuring the Quantum Leap will be available to consumers by the fourth quarter of this year."

* **Target (Summary):**
    > "San Jose-based Innovate Inc. has unveiled its 'Quantum Leap' processor, which is reportedly twice as fast and more power-efficient, with products expected to ship by Q4."

##### **Use Case Scenario**
The goal is to generate a short, coherent, and accurate summary from a longer document, capturing the most important information.

* **Executive Briefings:** A financial analyst feeds a 30-page quarterly earnings report into the system. The LLM outputs a one-paragraph executive summary highlighting key metrics like revenue, profit, and future outlook for a quick review by executives.
* **News Aggregation:** Summarizing multiple news articles about the same event to provide a comprehensive overview.
* **Meeting Productivity:** Condensing a one-hour meeting transcript into a short list of key decisions and action items.
* **Scientific Research:** Generating an abstract for a long scientific paper to help researchers quickly grasp its findings.

---
##### **How It Works:**
Text summarization is a classic sequence-to-sequence (seq2seq) task. It can be tackled effectively by two main architectural approaches.

#### Approach 1: Encoder-Decoder Models (The Classic Approach)
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
---
#### Approach 2: Decoder-Only Models (The Modern Approach)
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

---
#### Handling Long Documents During Training
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

---
#### Handling Long Documents During Inference

During inference, we only have the long document and need to generate a single, coherent summary. Since the model can't see the whole document, we use chunking-based strategies.

##### Strategy 1: Map-Reduce (Chunk, Summarize, Summarize Again)

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

##### Strategy 2: Iterative Refinement

This method attempts to maintain a running context as it moves through the document.

1.  Summarize the first chunk (`chunk_1`). Let's call the result `summary_1`.
2.  Take `summary_1`, append the next chunk (`chunk_2`) to it, and feed it to the LLM with a prompt like: `"Given the following summary so far and the next part of the document, refine and update the summary: Summary: <summary_1> Next Part: <chunk_2>"`.
3.  The model produces a new, updated summary (`summary_2`).
4.  Repeat this process, feeding `summary_n` and `chunk_n+1` into the model until all chunks have been processed.

* **Pros:** Can capture cross-chunk relationships better than Map-Reduce.
* **Cons:** It's a sequential process and therefore much slower. It can also suffer from "recency bias," where information from later chunks is weighted more heavily.

#### The Future: Native Long-Context Models

The strategies above are clever workarounds for a fundamental architectural limitation. The ultimate solution, which is rapidly becoming a reality, is to use models with much larger context windows.

Models like **Google's Gemini 1.5 Pro (with up to 1 million tokens)**, Anthropic's Claude series (100k-200k tokens), and specialized research models (Longformer, BigBird) are designed to handle much longer sequences.

For these models, you may not need a complex chunking strategy at all. You might be able to feed the entire document directly into the model's context window, allowing it to perform summarization with a full, holistic understanding of the text. This is the direction the field is moving.

***
