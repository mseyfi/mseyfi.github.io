[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/FineTuning-Comprehensive_Tutorial_on_Finetuning_LLMs-orange?style=for-the-badge&logo=github)](../../FineTuning)

# Tutorial: Instruction Fine-Tuning (SFT)

A base LLM, pre-trained on the vastness of the internet, is an incredible text completion engine. If you give it the prompt "The first person to walk on the moon was," it will expertly complete it with "Neil Armstrong."

However, if you ask it a direct question, `"Who was the first person to walk on the moon?"`, a base model might just continue the sentence with something like, `"and what did they say?"`. It doesn't inherently understand the *intent* to have a question answered. Instruction Fine-Tuning (SFT) teaches the model to understand and follow these instructions.

#### **Example Data**
The core of SFT is a high-quality dataset of instruction-response pairs. These pairs demonstrate the desired behavior to the model. The data can cover a wide range of tasks.

* **Simple Q&A:**
    * **Instruction:** `"What is the distance between the Earth and the Moon?"`
    * **Response:** `"The average distance between the Earth and the Moon is about 238,855 miles (384,400 kilometers)."`

* **Summarization:**
    * **Instruction:** `"Summarize this article: [long article text]"`
    * **Response:** `"[short, coherent summary of the article]"`

* **Creative Writing:**
    * **Instruction:** `"Write a short poem about the city of San Jose from the perspective of a bird."`
    * **Response:** `"From cypress spire, a grid below, where silicon valleys brightly glow..."`

* **Code Generation:**
    * **Instruction:** `"Write a Python function to calculate a factorial."`
    * **Response:** `def factorial(n): ...`

#### **Use Case Scenario**
The goal of SFT is to create a general-purpose, helpful assistant that can perform a wide variety of tasks based on user commands. The primary use case is the creation of conversational AI and chatbots.

* **A user wants to plan a trip:**
    * **User Prompt:** `"Create a 3-day itinerary for a family trip to San Francisco, focusing on activities suitable for young children."`
    * **Instruction-Tuned LLM Output:** The model understands the complex request (3-day plan, family focus, specific location) and generates a structured itinerary, perhaps including the Exploratorium, Golden Gate Park, and the California Academy of Sciences, complete with descriptions and logical flow.

---
#### **How It Works: A Mini-Tutorial**
Instruction fine-tuning reframes the model's objective from "predicting the next random token" to "predicting the tokens that form a helpful response to a given instruction."

##### **1. The Architecture**
The underlying architecture is typically a **Decoder-Only LLM**. The process doesn't change the core model but adapts its weights to this new instruction-following behavior.

##### **2. The Data Preparation**
This is the most critical step. A diverse, high-quality dataset is curated, often with significant human effort, to cover thousands of potential tasks. Each `(instruction, response)` pair is then formatted into a single sequence using a consistent template.

A common template format might look like this:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction_text}

### Response:
{response_text}
```

For our factorial example, the final training string would be:
`"Below is an instruction... ### Instruction: Write a Python function to calculate a factorial. ### Response: def factorial(n): ..."`

##### **3. The Training Phase**
The model is trained on these formatted sequences using the same fundamental process as text generation.

1.  **Input:** The entire formatted string (`Instruction` + `Response`) is fed into the Decoder-Only LLM.
2.  **Masking:** A **Causal Mask** is used, meaning when predicting a token at any position, the model can only see the tokens that came before it.
3.  **Loss Function (The Key Insight):** The goal is to teach the model to generate the `Response` *given* the `Instruction`. Therefore, we only care about the model's performance when it's generating the response part. We use a **Masked Cross-Entropy Loss**.
    * For all the tokens that are part of the prompt template and the `Instruction`, their loss is "masked out" or ignored (set to zero). We don't need to teach the model how to write the question.
    * The Cross-Entropy Loss is **only calculated for the tokens in the `Response`**. The model is penalized if it fails to predict the correct tokens of the ideal answer, forcing it to learn the desired behavior.
4.  **Parameter Updates:** This process is usually performed using a Parameter-Efficient Fine-Tuning (PEFT) method like **LoRA**. This means the original model's weights are frozen, and only a small set of adapter weights are trained. This makes the process much more efficient and prevents "catastrophic forgetting," where the model might lose its general knowledge.

##### **4. The Inference Phase**
Once the model is instruction-tuned, using it is straightforward.

1.  **Prompt the Model:** A user provides a new instruction, which is formatted using the *same template* used in training, but with the `Response` section left empty.
    ```
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    What are the three largest cities in California by population?

    ### Response:
    ```
2.  **Generate Autoregressively:** The model takes this formatted text as an input prefix. Because it has been extensively trained on this pattern, it knows that its task is to generate the tokens that form a helpful answer. It will begin generating the response one token at a time, continuing until it determines the answer is complete and outputs an end-of-sequence token.
