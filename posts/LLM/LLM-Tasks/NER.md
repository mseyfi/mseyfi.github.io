[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Information Extraction, Named Entity Recognition (NER)

##### **Example Data**

The goal of NER is to assign a categorical label to each token (or word) in a sentence. The data format consists of a sentence and a corresponding sequence of labels.

A common labeling format is the **IOB2 scheme**:
* **`B-TYPE`**: The **B**eginning of an entity of a certain TYPE (e.g., `B-PER` for Person).
* **`I-TYPE`**: **I**nside an entity of a certain TYPE. Used for entities that span multiple tokens.
* **`O`**: **O**utside of any entity.

* **Input Text:** `"Barack Obama was born in Hawaii."`
* **Tokenized Input:** `["Barack", "Obama", "was", "born", "in", "Hawaii", "."]`
* **Target Labels:** `[B-PER, I-PER, O, O, O, B-LOC, O]`

##### **Use Case Scenario**

The goal is to identify and pull out structured data (entities like people, organizations, dates, etc.) from unstructured, plain text.

* **An HR system processing a resume:**
    * **Input:** `"Worked as a software engineer at Google from 2019 to 2022."`
    * **System Extracts:** `{"Job Title": "software engineer", "Organization": "Google", "Start Date": "2019", "End Date": "2022"}`
* **News analysis:** Scanning thousands of articles to identify which companies are mentioned in relation to which locations.
* **Medical records:** Automatically extracting patient names, prescribed medications, and diagnoses from a doctor's notes.
* 
---
##### **How It Works: A Mini-Tutorial (Encoder-Only)**
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
