[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)
## [![GenAI](https://img.shields.io/badge/FineTuning-Comprehensive_Tutorial_on_Finetuning_LLMs-orange?style=for-the-badge&logo=github)](../../FineTuning)

# Instruction Tuning/ Supervised Fine-Tuning (SFT)

A base Large Language Model (LLM), pre-trained on the vastness of the internet, is an incredible text completion engine. If you give it the prompt "The first person to walk on the moon was," it will expertly complete it with "Neil Armstrong."

However, its core objective is just to predict the next logical word. If you ask it a direct question, `"Who was the first person to walk on the moon?"`, a base model might just continue the sentence with something like, `"and what did they say?"`. It doesn't inherently understand the *instructional intent* to have a question answered.

**Instruction Fine-Tuning (SFT)** is the critical process that retrains a model on examples of commands and their desired responses. It teaches the model to shift its goal from simply "completing text" to "following instructions and being helpful," which is the key to creating modern AI assistants.

#### **Example Data: The Foundation of SFT**
The success of SFT depends on a high-quality, diverse dataset of instruction-response pairs that demonstrate the desired behaviors. The more varied the examples, the more versatile the resulting model will be.

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
    * **Response:**
        ```python
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)
        ```
* **Information Extraction with Context:**
    * **Instruction:** `"Based on the context provided, what are the main duties of a Data Scientist?"`
    * **Context:** `"A Data Scientist analyzes complex data to help a company make better decisions. Key responsibilities include data mining, using machine learning to build predictive models, and visualizing data to communicate findings to stakeholders."`
    * **Response:** `"A Data Scientist's main duties involve analyzing complex data, building predictive models using machine learning, and creating visualizations to present their findings."`

#### **Use Case Scenario**
The primary goal of SFT is to create a general-purpose, helpful assistant that can perform a wide variety of tasks based on user commands. The main application is in creating conversational AI and chatbots.

* **A user wants to plan a trip:**
    * **User Prompt:** `"Create a 3-day itinerary for a family trip to San Francisco, focusing on activities suitable for young children."`
    * **Instruction-Tuned LLM Output:** The model understands the complex request (3-day plan, family focus, specific location) and generates a structured itinerary, perhaps including the Exploratorium, Golden Gate Park, and the California Academy of Sciences, complete with descriptions and logical flow.

---

### How It Works: A Detailed Breakdown

#### 1. The Architecture
The model at the heart of SFT is almost always a **Decoder-Only Transformer**. These models, like those in the GPT, Llama, and Gemini families, are inherently designed for sequential text generation, making them perfect for generating answers autoregressively.

* **Core Mechanism:** The architecture consists of a stack of decoder blocks. Each block contains two primary sub-layers:
    1.  **Masked Multi-Head Self-Attention:** The "masked" or "causal" nature means that when predicting a token at a certain position, the model can only attend to (or "see") the tokens that came before it. This enforces a left-to-right generative process.
    2.  **Feed-Forward Neural Network:** A standard fully connected network that processes the output of the attention layer.

SFT does not change this core architecture. Instead, it adapts the billions of numerical weights *within* these layers to favor generating helpful responses over generic text completions.

#### 2. The Data Preparation and Prompt Templates
This is a critical step. Each `(instruction, response)` pair is formatted into a single string using a consistent **prompt template**. The model learns to recognize this template, which signals that it should behave in instruction-following mode.

A popular template (used by models like Alpaca) looks like this:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction_text}

### Response:
```
During training, the ground-truth `response_text` is appended to this template. During inference, the template stops after `### Response:`, and the model generates the rest.

#### 3. The Training Process

* **Masking and Tokens:**
    * **Special Tokens:** Modern models use special tokens to delineate roles, such as `<|user|>` and `<|assistant|>`, along with an end-of-sequence token (`<|eos|>`) to teach the model when to stop. These are built into the prompt template.
    * **Causal Masking:** This is inherent to the decoder model and ensures it cannot "cheat" by looking ahead at the response tokens while predicting them.

* **The Loss Function (and its Mathematics):**
    The goal is to teach the model to generate the `Response` *given* the `Instruction`. Therefore, we only care about the model's performance when it's generating the response part. This is achieved using a **Masked Cross-Entropy Loss**.

    1.  The model processes the entire formatted sequence (`Instruction` + `Response`).
    2.  At each position `t`, it outputs a vector of **logits**, representing a score for every possible token in the vocabulary.
    3.  A `softmax` function is applied to these logits to get a probability distribution, $P(x_t)$, over the vocabulary for the next token.
    4.  The standard **Cross-Entropy Loss** for a single token is the negative log-probability of the true next token, $y_t$.
        $$L_t = -\log P(y_t)$$
    5.  **The Masking:** We create a "loss mask" — a list of 1s and 0s. The mask value is `0` for all tokens in the `Instruction` and prompt template sections, and `1` for all tokens in the `Response` section.
    6.  The final loss for the entire sequence is the sum of the per-token losses, multiplied by the mask. This effectively zeroes out the loss for the instruction part.

    Mathematically, for a single training sequence with `T` tokens and a loss mask `m`, the total loss `L` is:
    $$L = \sum_{t=1}^{T} m_t \cdot (-\log P(y_t))$$
    Where $m_t = 0$ if token `t` is part of the instruction, and $m_t = 1$ if it's part of the response. This forces the model to focus all of its learning "effort" on generating the correct response.

#### 4. Sample Code (Conceptual Hugging Face TRL)
Training is often done with libraries like Hugging Face's `transformers` and `TRL` (Transformer Reinforcement Learning), using a `SFTTrainer`.

```python
# Conceptual Code for SFT
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load a base model and tokenizer
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load and format the instruction dataset
# This assumes your dataset has a 'text' column where each entry is a fully formatted prompt.
dataset = load_dataset("your_instruction_dataset.json", split="train")

# 3. Configure PEFT method (e.g., LoRA) for efficient training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. Configure Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500
)

# 5. Initialize the Trainer
# The SFTTrainer automatically handles the masked loss function for you.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="text", # The column in your dataset with the formatted prompt
)

# 6. Start Training
trainer.train()

# 7. Save the trained adapter
trainer.save_model("./final_adapter")
```

#### 5. Inference
After fine-tuning, you use the model by formatting your new instruction with the *exact same template*, leaving the response section empty.

```python
from transformers import pipeline

# Load the base model and merge the LoRA adapter
model_path = "./final_adapter"
pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer)

# Create the prompt using the template
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What are the main attractions in downtown San Jose, California?

### Response:
"""

# Generate the response
result = pipe(prompt_template, max_new_tokens=250)
print(result[0]['generated_text'])
```

---

### References and Nominal Models

* **Key Papers:**
    * **"Finetuned Language Models Are Zero-Shot Learners" (FLAN)** (Wei et al., 2021): A foundational paper showing that instruction-tuning on a massive mix of tasks dramatically improves zero-shot performance on unseen tasks.
    * **"Training language models to follow instructions with human feedback" (InstructGPT)** (Ouyang et al., 2022): The seminal paper from OpenAI that detailed the SFT and RLHF process used to create models that are helpful and aligned.
    * **"Self-Instruct: Aligning Language Models with Self-Generated Instructions"** (Wang et al., 2022): Introduced a method for using a powerful LLM to generate its own instruction-tuning data, bootstrapping the process.
* **Popular Open-Source Instruction-Tuned Models:**
    * **Llama-3-8B-Instruct & Llama-3-70B-Instruct:** State-of-the-art instruction-tuned models from Meta.
    * **Mistral-7B-Instruct:** A highly capable smaller model from Mistral AI, known for its efficiency.
    * **Gemma-7B-it:** Google's family of open models, with instruction-tuned variants.
    * **Dolly-v2-12b:** One of the first truly open-source, commercially viable instruction-following models, created by Databricks.
    * **Alpaca 7B:** An early and influential model from Stanford, fine-tuned from the original LLaMA model on data generated by GPT-3.
