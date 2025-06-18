[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../../main_page/GenAI)

[![GenAI](https://img.shields.io/badge/LLM_TASKs-Selected_LLM_TASKs-orange?style=for-the-badge&logo=github)](../LLM-Tasks)

# Reasoning


##### **Example Data**
The key to teaching reasoning is the data format. Instead of just a question and a final answer, the target data includes the intermediate thinking steps. This is known as Chain of Thought (CoT).

* **Input (Question):**
    > `"If John has 5 apples and gives 2 to Mary, how many does he have left?"`

* **Target (Chain of Thought + Answer):**
    > `"[REASONING] John starts with 5 apples. He gives away 2 apples. To find out how many are left, we need to subtract the number of apples given away from the starting amount. The calculation is 5 - 2. [REASONING] 5 - 2 = 3. [ANSWER] The final answer is 3."`

##### **Use Case Scenario**
The goal is to solve problems that cannot be answered with a simple fact, but require logical, arithmetic, or commonsense steps.

* **Multi-step Math Problems:** A user prompts the model with a classic word problem from a local perspective:
    > `"A Caltrain leaves the San Jose Diridon station at 10:00 AM traveling north at 60 mph. A car leaves the same station at 11:00 AM, following the same route at 70 mph. At what time will the car catch up to the train?"`
* **The LLM provides a step-by-step breakdown:** It first calculates the train's one-hour head start (60 miles). Then it finds the relative speed of the car (10 mph). Finally, it divides the distance by the relative speed to find the time taken (6 hours) and calculates the final time (5:00 PM).
* **Other Uses:** Solving logic puzzles, debugging code by reasoning about the error, and planning complex strategies.

---
##### **How It Works:**
Reasoning is not a feature that is explicitly programmed into LLMs. It is an **emergent capability** of very large models, significantly enhanced by a technique called Chain of Thought (CoT) fine-tuning.

##### **What is an "Emergent Capability"?**
An emergent capability is a behavior that appears in large models that was not present in smaller models. It arises spontaneously from the sheer scale of the model's parameters and training data. The simple task of "predicting the next token," when performed over trillions of words, leads to the model learning complex underlying patterns of logic and reasoning.

##### **The Key Technique: Chain of Thought (CoT)**
The core idea behind CoT is simple but powerful: **forcing the model to "show its work."**

When a model tries to jump directly to a final answer for a complex problem, it's more likely to make a mistake. By fine-tuning the model to first generate the step-by-step reasoning and *then* the final answer, we teach it a more robust and reliable problem-solving process.

##### **Architecture**
Reasoning is an advanced sequential generation task, making it the domain of large **Decoder-Only models** like Google's Gemini, GPT-4, and Llama.

##### **The Training Phase (CoT Fine-Tuning) ✍️**

1.  **The Data:** The training data consists of pairs of `(Question, Full_CoT_Answer)`. This data is often meticulously created by humans to teach the model high-quality reasoning patterns.
2.  **Input Formatting:** The question and the full target (reasoning + answer) are formatted into a single continuous sequence.
3.  **Architecture & Loss Function:**
    * The model is a standard **decoder-only** architecture using **Causal Masking**.
    * The loss function is **Cross-Entropy Loss**, and this is the crucial part: the loss is calculated over the **ENTIRE target sequence**.
    * This means the model is penalized not just for getting the final answer `3` wrong, but for getting any token in the reasoning steps (e.g., `"subtract"`, `"-"`, `"="`) wrong. This forces the model to learn the *process* of logical deduction, not just to memorize final answers.

##### **The Inference Phase (Solving a New Problem) 🧠**

1.  **The Prompt:** The user provides only the question (e.g., the train problem).
2.  **The Generation Loop:** The model takes the question as its initial prompt and begins to generate the solution autoregressively.
3.  **The Autoregressive Process:**
    * Because the model has been trained to output reasoning first, the highest probability next tokens will naturally form the step-by-step thinking process. It doesn't jump to the answer because that's not the pattern it learned.
    * It generates token by token (`predict -> sample -> append`), laying out its entire chain of thought before concluding with the final answer.
    * For reasoning tasks, the sampling "temperature" is often set very low (close to 0) to make the output more deterministic and logically consistent.

##### **A Note on Few-Shot CoT Prompting**
For the most powerful models, you may not even need to fine-tune them. You can elicit reasoning behavior directly in the prompt by providing an example. This is called **few-shot prompting**.

**Example:**
```
Q: If John has 5 apples and gives 2 to Mary, how many does he have left?
A: John starts with 5 apples. He gives away 2. 5 - 2 = 3. The final answer is 3.

Q: If a train leaves San Jose at 10:00 AM traveling at 60 mph... at what time will the car catch up?
A: [The model will now generate the step-by-step reasoning because it follows the format of the example provided.]
```
---
Of course. Here is a comprehensive mini-tutorial on Sentence-Transformers that synthesizes our entire discussion, including the architecture, use cases, different training philosophies, and the mathematical details of the loss functions.

***

