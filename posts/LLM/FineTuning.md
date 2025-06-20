[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# Fine-Tuning Large Language Models
------

Fine-tuning Large Language Models (LLMs) can be approached in several ways, each balancing compute, memory, data efficiency, and performance. The major categories are:

---

## **1. Full Fine-Tuning**

* **What it is**: Update **all parameters** of the model using backpropagation on task-specific data.
* **Use case**: When you have **a lot of labeled data** and **enough compute** (e.g., cloud TPU/GPU clusters).
* **Pros**: Best performance; full model adapts.
* **Cons**: Extremely resource-intensive (memory, compute, storage).
* **Example**: Fine-tuning GPT-2 or BERT on a specific domain corpus.

---

##  **2. Parameter-Efficient Fine-Tuning (PEFT)**

Only a **small subset** of parameters are updated. Popular PEFT methods include:

## [![adaptor](https://img.shields.io/badge/Adaptor_Tuning-Adapter_based_Fine_Tuning-blue?style=for-the-badge&logo=github)](FineTuning/Adaptor-Finetuning)

* Inject **adapter modules** (small bottleneck layers) between transformer layers.
* Only adapters are trained.
---

## [![LORA](https://img.shields.io/badge/LORA-Low_Rank_Adaptation-blue?style=for-the-badge&logo=github)](../posts/LLM/FineTuning/LORA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large-scale pre-trained models. It allows us to adapt a model by introducing low-rank trainable matrices into certain parts of the network while keeping the original pre-trained weights frozen. <p></p>
</div>

## [![prefix](https://img.shields.io/badge/Prefix_Tuning-LOW_RANK_ADaptation-blue?style=for-the-badge&logo=github)](FineTuning/Prefix-Tuning)

* Prepend learnable "prefix vectors" to attention layers (not the input).
* Doesn’t change model weights.
* Efficient for long-context tasks.
---
## [![ptune](https://img.shields.io/badge/P--Tuning--V2-Deep_Prompt_Tuning-blue?style=for-the-badge&logo=github)](FineTuning/P-Tuning-V2)

* Learn **soft prompt embeddings** (continuous vectors) fed into model’s input.
* Can also involve updating a small MLP.

---

## 🧷 **3. Prompt Tuning / Soft Prompts**

* Learn only a **small set of embedding vectors** (the "prompt") prepended to the input.
* Can be:

  * **Discrete** (crafted text)
  * **Soft** (learned embeddings)
* **No model weight change**, only optimize embeddings.
* Good for low-resource tasks.
* **Example**: [Lester et al. 2021 – "The Power of Scale for Parameter-Efficient Prompt Tuning"](https://arxiv.org/abs/2104.08691)

---

## [![SFT](https://img.shields.io/badge/SFT-Instruction_Fine_Tuning(SFT)-blue?style=for-the-badge&logo=github)](FineTuning/SFT)
* Finetune on (instruction, response) pairs.
* Trains LLMs to follow human-written instructions.
* Often a precursor to RLHF.
* **Example**: FLAN-T5, Alpaca, OpenAssistant, etc.  
---

## 🧑‍⚖️ **5. Reinforcement Learning with Human Feedback (RLHF)**

* Three stages:

  1. **SFT**: Train on instruction-following pairs.
  2. **Reward model**: Trained to rank outputs.
  3. **PPO**: Optimize model to maximize reward signal.
* Used in **ChatGPT, Claude, Gemini, etc.**

---

## 💾 **6. Domain-Adaptive Pretraining (DAPT)**

* Continue **unsupervised pretraining** on in-domain text.
* No labels required.
* Useful for **medical, legal, code** domains.

---

## 🌱 **7. Multi-Task Fine-Tuning**

* Fine-tune on **multiple tasks simultaneously**.
* Model generalizes better across domains and instructions.
* Example: T5, FLAN.

---

## 🧬 **8. Quantized/Low-Precision Finetuning**

* Combine LoRA or adapters with **quantized models** (e.g., 4-bit QLoRA).
* Enables fine-tuning 65B+ models on consumer GPUs.
* Example: **QLoRA**, **GPTQ + LoRA**.

---

## 🧱 Summary Table

| Method                   | Params Updated   | Compute Cost | Memory   | Use Case                                |
| ------------------------ | ---------------- | ------------ | -------- | --------------------------------------- |
| Full Fine-Tuning         | All              | 🔥🔥🔥       | 🔥🔥🔥   | High-resource, high-accuracy            |
| Adapter Tuning           | \~1–5%           | 🔥           | 🔥       | Modular fine-tuning                     |
| LoRA                     | \~0.1–1%         | 💡           | 💡       | Most popular for cost-effective tuning  |
| Prefix/Prompt Tuning     | \~0.01–1%        | 💡           | 💡       | Efficient for low-resource tasks        |
| Instruction Tuning (SFT) | All or PEFT      | 🔥 or 💡     | 🔥 or 💡 | Aligns model to human instructions      |
| RLHF                     | All (multi-step) | 🔥🔥🔥       | 🔥🔥🔥   | Chatbots, assistant models              |
| DAPT                     | All              | 🔥           | 🔥       | In-domain generalization without labels |
| Multi-task Finetuning    | All or PEFT      | 🔥 or 💡     | 🔥 or 💡 | Improves robustness across tasks        |
| QLoRA                    | 0.1–1%, 4-bit    | 💡           | 💡       | Finetuning large models on single GPU   |









### 3. Parameter-Efficient Fine-Tuning (PEFT)

PEFT is a family of techniques born from the question: "Why update billions of parameters when you can get almost the same result by updating just a tiny fraction?" The core idea is to **freeze the massive pre-trained model** and only train a small number of new, added parameters. This is like adding small, lightweight attachments to a large, powerful engine instead of rebuilding the engine itself.



## [![prmt](https://img.shields.io/badge/Prompt_Tunint-Prompt_Tuning/Prefix_Tuning-blue?style=for-the-badge&logo=github)](FineTuning/Prompt)

This is another PEFT method that takes a different approach. Instead of modifying the model, it modifies the input.

- **The Goal:** To steer the behavior of a completely frozen model by learning an optimal "prompt."

- How It Works:

   The entire LLM is frozen. A small sequence of special tokens (a "soft prompt" or "prefix") is added to the beginning of the input. 

  During training, only the embedding vectors for these special prefix tokens are updated.19

   The model learns the perfect "magic words" to prepend to any user input to guide the frozen LLM into producing the desired output for a specific task.

- **What is Trained:** Only the prefix embeddings (often just a few thousand parameters).

- **Pros:** The most parameter-efficient method.

- **Cons:** Can be less powerful than LoRA because it has less influence over the model's internal computations.

------

## [![RLHF](https://img.shields.io/badge/RLHHF-Reinforcement_Learning_from_Human_Feedback-blue?style=for-the-badge&logo=github)](FineTuning/RLHF)

RLHF is an advanced fine-tuning stage that comes *after* Instruction Fine-Tuning.20 Its goal is not to teach the model new knowledge, but to align its behavior with complex human values like helpfulness, harmlessness, and honesty.



- **The Goal:** To make a model more preferable and safer to interact with.

- How It Works (Simplified):

  1. **Collect Data:** Prompt the instruction-tuned model (SFT model) to generate several different answers to a single prompt. A human then ranks these answers from best to worst.

  2. **Train a Reward Model:** Train a separate model (the "reward model") on this data. Its job is to learn to predict which responses humans will prefer. It learns to output a score for any given response.

  3. Fine-tune with RL:

      Use a reinforcement learning algorithm (like PPO) to continue fine-tuning the SFT model. 

     The model generates a response, the reward model scores it, and this score is used as the "reward" to update the LLM's parameters.21

     The LLM gets "points" for generating responses that the reward model thinks a human would like.22

- What is Trained:

  The LLM's parameters are updated via reinforcement learning, and a separate reward model is also trained.23

- **Pros:** The most effective known method for reducing harmful outputs and aligning models with human preferences.

- **Cons:** A highly complex and expensive process requiring significant data collection and multiple training loops.
