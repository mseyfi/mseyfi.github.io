[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Retrieval Augmented Generation (RAG):

The core idea behind RAG is to solve a major weakness of LLMs: they can only answer based on the data they were trained on, which might be outdated or not include private information. RAG gives the LLM an "open book" to use when answering a question, ensuring the response is timely, accurate, and grounded in specific facts. It transforms a "closed-book" memory test into an "open-book" reasoning test.

##### **Example Data & Components**
RAG is a system composed of several parts, not a simple `Input -> Target` training pair.

1.  **The Knowledge Base (The "Book"):** A collection of private or specific documents.
    * *Document 1:* "San Jose City Council Meeting Minutes - June 10, 2025.pdf"
    * *Document 2:* "Urban Sky Project Details.docx"
    * *Document 3:* "Local Business Impact Report - June 2025.pdf"

2.  **The User Query (The "Question"):** A question that can only be answered using the knowledge base.
    > "What was the city council's final decision on the 'Urban Sky' zoning proposal in San Jose?"

3.  **The Final LLM Output (The "Answer"):** A synthesized answer based on the retrieved information.
    > "Based on the meeting minutes from June 10, 2025, the San Jose City Council approved the 'Urban Sky' zoning proposal with a 7-4 majority."

##### **Use Case Scenario**
The goal of RAG is to ground a powerful generative model in a specific, up-to-date, or private set of documents, making its answers more trustworthy and factual.

* **Enterprise Chatbot:** An employee asks an internal chatbot, `"What is our corporate policy on work-from-home for 2025?"` The RAG system retrieves the latest HR document from the company's internal server and provides the current, correct answer.
* **Up-to-Date Customer Support:** A customer asks a support bot, `"Is the new product compatible with my device?"` The RAG system pulls the latest product manuals to provide an accurate technical answer, even if the product was released after the LLM was trained.

---
##### **How It Works: A Mini-Tutorial**
RAG is primarily an **inference-time** process that involves two main phases: preparing the knowledge base (Indexing) and answering the query (Retrieval & Generation).

##### **Phase 1: The Indexing Phase (Offline Preparation - "Building the Library")**
This is a one-time setup process you perform on your knowledge base to make it searchable.

1.  **Load & Chunk Documents:** Your system ingests your source documents and breaks them down into smaller, manageable chunks (e.g., paragraphs). This is crucial because a small, focused chunk is more likely to match a specific question than a whole document.

2.  **Embed Chunks:** Each chunk of text is passed through an **embedding model** (like a Sentence-Transformer). This model converts the text into a numerical vector (an embedding) that captures its semantic meaning.

3.  **Store & Index Chunks:** This is where the system gets clever. The embeddings and their corresponding text are stored.
    * **How we retrieve the context from the embedding:** You are right to question how this link is maintained. We do not store the text *inside* the embedding. Instead, two components are used together:
        1.  A **Vector Index** (using a library like **FAISS**): This stores the numerical embedding vectors and is highly optimized for finding the most similar vectors to a query vector.
        2.  A **Document/Metadata Store** (like a dictionary, hash map, or key-value database): This store maps a unique ID for each chunk to its original text content. The Vector Index also stores this ID alongside the embedding.

    * **Efficient Searching with FAISS:** For millions of chunks, comparing a query to every single chunk is too slow (brute-force search). **FAISS (Facebook AI Similarity Search)** is a library that builds an efficient index to speed this up. A common technique it uses is an **Inverted File Index (IVF)**:
        * **Clustering:** FAISS first uses a clustering algorithm (like k-Means) to group all the document chunk embeddings into, for example, 1000 clusters. Each cluster has a "centroid" representing its center.
        * **Partitioning:** The vector space is now partitioned into 1000 regions (or "cells").
        * **Efficient Search:** When a new query comes in, FAISS first compares the query vector to just the 1000 cluster centroids. Instead of searching all one million chunks, it only searches within the most promising clusters (e.g., the top 5 closest clusters), drastically reducing search time. This is an example of Approximate Nearest Neighbor (ANN) search.

##### **Phase 2: The Retrieval & Generation Phase (Real-time Inference)**
This happens every time a user asks a question.

1.  **Receive and Embed User Query:** The user's question is converted into a query vector using the *same* embedding model.
2.  **Retrieve Relevant Chunk IDs:** The system uses the query vector to search the FAISS index. The index rapidly returns the **IDs** of the `top-k` (e.g., top 5) most semantically similar document chunks.
3.  **Lookup a The Chunks:** The system takes these retrieved IDs and looks them up in the Document Store (the hash map/key-value store) to get the original text of those chunks. This is the **RETRIEVAL** step.
4.  **Augment the Prompt:** A new, detailed prompt is constructed for the generative LLM, combining the retrieved context with the original user question.
    ```bash
    Prompt Template:
    
    "Using ONLY the following context, please answer the user's question. If the answer is not in the context, say you don't know.
    
    Context:
    <Retrieved Chunk 1 text>
    <Retrieved Chunk 2 text>
    
    Question:
    <Original User Question>"
    ```
5.  **Generate the Answer:** This augmented prompt is sent to a powerful generative LLM (like Gemini). The LLM synthesizes a final answer based *only* on the provided information. This is the **GENERATION** step.
***
