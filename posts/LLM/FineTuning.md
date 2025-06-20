[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# Fine-Tuning Large Language Models
------

Fine-tuning Large Language Models (LLMs) can be approached in several ways, each balancing compute, memory, data efficiency, and performance. The major categories are:

---

## **1. General (All/PEFT) Fine-Tuning**
## [![SFT](https://img.shields.io/badge/SFT-Instruction_Fine_Tuning(SFT)-blue?style=for-the-badge&logo=github)](FineTuning/SFT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Instruction Fine-Tuning/Supervised Fine-Tuning (SFT) is the critical process that retrains a model on examples of commands and their desired responses. It teaches the model to shift its goal from simply "completing text" to "following instructions and being helpful," which is the key to creating modern AI assistants.<p></p>
</div>

##  **2. Parameter-Efficient Fine-Tuning (PEFT)**

Only a **small subset** of parameters are updated. Popular PEFT methods include:

## [![adaptor](https://img.shields.io/badge/Adaptor_Tuning-Adapter_based_Fine_Tuning-blue?style=for-the-badge&logo=github)](FineTuning/Adaptor-Finetuning)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
As Large Language Models (LLMs) grew to billions of parameters, Full Fine-Tuning—updating every single weight for each new task—became prohibitively expensive. It required massive computational resources and resulted in a new, full-sized model copy for every task.

Adapter Tuning was proposed as a solution. The core idea is simple yet profound: what if we could freeze the massive pre-trained LLM, which already contains vast general knowledge, and only train a handful of tiny, new parameters for each specific task? 
<p></p>
</div>


## [![LORA](https://img.shields.io/badge/LORA-Low_Rank_Adaptation-blue?style=for-the-badge&logo=github)](../posts/LLM/FineTune/LORA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large-scale pre-trained models. It allows us to adapt a model by introducing low-rank trainable matrices into certain parts of the network while keeping the original pre-trained weights frozen. <p></p>
</div>

## [![prefix](https://img.shields.io/badge/Prefix_Tuning-LOW_RANK_ADaptation-blue?style=for-the-badge&logo=github)](FineTuning/Prefix-Tuning)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
In the landscape of Parameter-Efficient Fine-Tuning (PEFT), methods like Adapter Tuning modify the model's architecture by injecting new layers. Prefix-Tuning proposes an even less invasive idea: what if we could achieve specialized behavior without touching the model's architecture at all?

The core idea is to freeze the entire pre-trained LLM and learn a small sequence of special, continuous vectors, a "prefix", that we prepend to the input. This learned prefix acts as an optimized set of instructions that steers the frozen LLM's attention and directs it to perform the desired task.<p></p>
</div>

## [![ptune](https://img.shields.io/badge/Prompt--Tuning-Prompt_Tuning/Soft_Prompts-blue?style=for-the-badge&logo=github)](FineTuning/Prompt-Tuning)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Prefix-Tuning was a powerful idea: steer a frozen LLM by learning continuous "virtual tokens" that are prepended to the keys and values in every attention layer. However, it had some challenges. The training could sometimes be unstable, and its performance wasn't always as strong as full fine-tuning on harder, smaller-scale datasets.

<b>Prompt-Tuning</b> was developed to address these issues. It adopts the core concept of using continuous prompts at the input layer.<p></p>
</div>

## [![ptunev2](https://img.shields.io/badge/P--Tuning--V2-Deep_Prompt_Tuning-blue?style=for-the-badge&logo=github)](FineTuning/P-Tuning-V2)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
<b>P-Tuning v2</b> introduces Deep Prompt Tuning. Instead of one instruction at the start, you place a small, expert guide on every single floor of the building. This guide provides continuous, layer-specific instructions to each department, ensuring the final output is exactly what you want. It's a continuous conversation with the model, not a single whisper at the start.

This deep guidance is why <b>P-Tuning v2</b> is more powerful, stable across all model sizes, and excels at the complex tasks where shallow methods fail.<p></p>
</div>

## **3. Reinforcement Learning with Human Feedback (RLHF)**

## [![RLHF](https://img.shields.io/badge/RLHHF-Reinforcement_Learning_from_Human_Feedback-blue?style=for-the-badge&logo=github)](FineTuning/RLHF)



## **4. Domain-Adaptive Pretraining (DAPT)**

* Continue **unsupervised pretraining** on in-domain text.
* No labels required.
* Useful for **medical, legal, code** domains.

---

## **5. Multi-Task Fine-Tuning**

* Fine-tune on **multiple tasks simultaneously**.
* Model generalizes better across domains and instructions.
* Example: T5, FLAN.

---

## **6. Quantized/Low-Precision Finetuning**

* Combine LoRA or adapters with **quantized models** (e.g., 4-bit QLoRA).
* Enables fine-tuning 65B+ models on consumer GPUs.
* Example: **QLoRA**, **GPTQ + LoRA**.

---

## Summary Table

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
