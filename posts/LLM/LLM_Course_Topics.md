# Big Topics in LLMs

1- what is Tokenizer, what are different tokenizers?

2- Embedding 

3- How de-tokenizatin

4-Transformer architechtures:
  - Encoder only models (BERT) sequence-to-sequence, (input output same size) or use case: sentiment, classification
  - Encoder-decoder models sequence-to-sequence with different length (BArt, T5)
  - Decoder only architechtures (Jurassic, Lama, GPT, BLUE)


- text you enter to the model is prompt
- output of the model is completion

- Prompt ---> inference ----> completion
- context window: full amount of text available to memory

5- In context Learning and zero/one/few shot inference

6- greedy(select the word according to the softmax highest probability) vs random sampling text generation (select a text randomly based on the softmax probability- if a word has 2% chance it can still be selected however in the greedy case there is no way a word with less maximum softmax probability be chosen)

7- top-k and top-p sampling and temperature

8- Encoder only models (BERT, ROBERTA) Autoencoding models ,  
  - masked languege modeling (MLM)
  - mask a word and predict a word (self supervisedly)
  - use bidirectional context, meaning both the past and the future words can take part to predicting the missing word.  - 
  - used for Named Entity Recognition)
  - Sentiment analysis
  - token/sentence classification
    -BERT and ROBERTA

9- Decoder only (GPT, Lamma) or autoregressive models (Causal Language modeling ) --> Text prediction from previsous tokens. 
The model has no knowledge of the end of the sentence.Unidirectional context, very good in zero shot inference

10- Encoder-Decoder models(T5, Bart) summarization, translation, question answering   
  - pretrain using span corruption
  - Sentinel tokens
# Computational Challenges of LLMs
BFLOAT16 and FP16
Model Sharding FSDP
Chinchilla Rule 
## Finetuning

ICL (in-contect learning zeor/one/few shot learning doesnt work for smaller models and may take a lot of space in the context window)

1- Instruction fine-tuning 
  - works on pairs of prompt-completion.
  - For a single task fine-tuning we need around 1K examples.
  - it's a full fine-tuning mechanism that updates all the model weights
  - Finetuning on a single task may cause catastrophic forgetting
  - If model performance drop on other tasks are important, you need to finetune over multiple tasks
    - Often 50K to 100K examples across multiple taks are needed.
  - **PEFT** or parameter efficient fine-tuning only adapt task-specific weights.
  - One important technique to prevent catasrophic forgetting is regularization on gradients norms. Using this technique, we prevent big updates on the backpropagation.
  - ## FLAN Finetuned LAnguage NET.
2- Metrics for evaluating LLMS
  - ## Definitions:
    -  **Unigram** is a single word  
    -  **Bigram** is a two words
    -  **n-gram** is a group of n *consecutive* words
  
  - Rouge: Used for text summarization
    - compares a summary to one or multiple other reference summaries. 
    
    EX:

    Reference: It is cold outside
    
    Generated output: it is very cols outside 
    
    $$
    Rouge-1_{\text{recall}} = \frac{\text{unigram matches}}{\text{unnigrams in reference}}
    $$

    
    $$
    Rouge-1_{\text{precision}} = \frac{\text{unigram matches}}{\text{unnigrams in output}}
    $$

    $$
    Rouge-1_{\text{F1}} = 2.\frac{\text{precision}\times \text{recall}}{\text{precision} + \text{recall}}
    $$
    
  - Issues: ordering of words are not taken into consideration therefore Rouge_1 can have a high number but they results may subjectively be poor.
  - We can use N-grams instead of word matching for Rouge
  - **LCS**: Longest common sequence
  - BLEU: Used for translation task
    - Compares translation to human-generated translation   

  # RLHF
-  Reward
-  Rollout
-  state and context
-  scaling instruction-finetuned language models
-  Convert ranking into pariwise training data for the reward model
- Reorder the prompts so that the prefered option comes first 

 ## Why Convert ranking into pariwise training data for the reward model

  Learning from full rankings (e.g., A > B > C > D) is more **complex and less stable** than learning from pairwise preferences due to several intertwined mathematical, computational, and statistical reasons. Here’s a breakdown:

---

### 📌 1. **Combinatorial Explosion of Permutations**

* A full ranking is a **permutation** over $n$ items.
* The number of possible permutations grows **factorially**: $n!$
* Modeling or assigning probabilities over all possible permutations (as in the **Plackett-Luce model**) becomes intractable for larger $n$.

#### 🔢 Example:

* For 4 items: 24 possible rankings.
* For 10 items: 3,628,800 possible rankings.

This makes the **output space huge**, and **gradient updates** less localized.

---

### 📌 2. **Label Ambiguity and Noise Amplification**

* Human rankings are **subjective and noisy**, especially as the number of choices increases.
* Annotators are less consistent with full rankings:

  * A > B might be confidently judged.
  * But deciding whether C > D is often weakly supported.

#### ✅ Pairwise comparisons focus only on two items at a time:

* Less cognitive load
* Easier and more accurate judgment
* Reduced label noise

---

### 📌 3. **Loss Function Instability**

* To learn from full rankings, we often use **listwise losses** such as:

  * ListMLE
  * Plackett-Luce
  * ListNet

These losses involve **normalizations over permutations** and are:

* **Non-convex**
* **Sensitive to small changes** in ranking order
* Difficult to optimize with SGD
* Require special sorting operations that **break gradient flow**

---

### 📌 4. **Difficulty in Generalizing**

* Ranking models often need to **predict the entire order** of a new list, not just compare two items.
* This task is **overconstrained**:

  * A small mistake (e.g., swapping B and C) drastically affects listwise metrics like NDCG, MAP.
  * Pairwise models are more forgiving and often generalize better under human-labeled data.

---

### 📌 5. **Sparse Learning Signal**

* A full ranking provides **only one ground truth permutation**.
* But from the same data, you can extract $\binom{n}{2}$ pairwise preferences.
* Pairwise learning gives **many training samples per annotation**, which helps generalization and convergence.

---

### 🔁 Comparison Table

| Aspect              | Full Ranking (Listwise)                 | Pairwise                     |
| ------------------- | --------------------------------------- | ---------------------------- |
| Complexity          | $O(n!)$                                 | $O(n^2)$                     |
| Labeling difficulty | High (subjective, error-prone)          | Low (binary comparison)      |
| Learning signal     | Sparse (1 permutation)                  | Dense (many comparisons)     |
| Optimization        | Complex loss, non-convex                | Simple sigmoid or hinge loss |
| Gradient stability  | Poor (discontinuous at sort boundaries) | Smooth (score differences)   |
| Scalability         | Poor for large $n$                      | Good                         |

---

### 🧠 Intuition

Training with full rankings is like trying to learn the rules of chess by memorizing full games. Training with pairwise comparisons is like learning which moves are better than others—simpler and more generalizable.



 ## reorder prompts so the preferred response comes first

In **RLHF reward model training**, we **reorder prompts so the preferred response comes first** (e.g., change (bad, good) → (good, bad)) before feeding them into the model. Here's **why**:



#### ✅ 1. **Standardization for Loss Computation**

Most reward models use a **pairwise loss** like:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r(x_{\text{preferred}}) - r(x_{\text{dispreferred}}))
$$

To compute this, you need to know **which input is preferred**. By always putting the preferred item first, the model:

* Always computes $r_1 - r_2$ (where $r_1 = r(x_{\text{preferred}})$)
* Uses the same loss formula every time
* Avoids needing if-else logic or label-dependent signs

This is critical for clean and **vectorized implementation**.

---

#### ✅ 2. **Avoids Label Ambiguity**

If you feed unordered pairs with an extra label (`label = 0` means "first is better", `label = 1` means "second is better"), then:

* Your implementation must **flip** the sign of the reward difference based on the label
* Easy to introduce bugs like sign errors or flipped labels

Reordering the preferred item first removes this ambiguity.

---

#### ✅ 3. **Eases Batch Processing**

If all examples are ordered (preferred, dispreferred), then:

* You can construct two input batches:

  * `x_winners = [x1, x2, x3, ...]`
  * `x_losers  = [y1, y2, y3, ...]`
* Compute scores:

  $$
  r_{\text{win}} = r(x_winners),\quad r_{\text{lose}} = r(x_losers)
  $$
* Vectorized loss:

  $$
  \mathcal{L} = -\log \sigma(r_{\text{win}} - r_{\text{lose}})
  $$

---

#### ✅ 4. **Simplifies Human Feedback Integration**

When collecting rankings or binary preferences from humans, you only need to mark one response as preferred. Reordering ensures that training data **always places the good one first**, allowing clean downstream usage.

---

### 🧠 Summary

| Why Reorder?  | Benefit                                              |
| ------------- | ---------------------------------------------------- |
| Standard loss | Consistent $\log \sigma(r_1 - r_2)$ without flipping |
| Avoids bugs   | No need to handle label-dependent signs              |
| Vectorization | Enables batch training with simple subtraction       |
| Clean design  | Better integration with data pipelines and logging   |

---

Absolutely. Let’s now **rewrite everything using BERT** as the backbone of the reward model, and **track all tensor shapes step-by-step**, from input construction to loss computation.

---

# 🎯 Reward Model Using BERT – Full Tutorial with Tensor Sizes

---

## 1. 📥 **Inputs**

We start with:

* A prompt $x$
* Two completions $y^+$ (preferred), $y^-$ (dispreferred)

### Example:

```text
Prompt:     "Tell me a joke."
Preferred:  "Why don’t scientists trust atoms? Because they make up everything."
Dispreferred: "I'm not sure."
```

### Input Construction:

We concatenate prompt and completion with a separator:

```text
[CLS] Tell me a joke. [SEP] Why don’t scientists trust atoms? Because they make up everything. [SEP]
```

### Tokenization:

Assume BERT-base:

* Vocabulary size: $V = 30522$
* Hidden size: $H = 768$
* Max input length: $L = 512$

Let’s assume:

* Prompt + preferred response = 40 tokens
* Prompt + dispreferred response = 20 tokens

So:

```python
input_ids_preferred:    [batch_size, 40]
input_ids_dispreferred: [batch_size, 20]
```

After padding to max length in batch (say, 40), both inputs become:

```
input_ids: [batch_size, seq_len] = [B, 40]
attention_mask: [B, 40]
token_type_ids: [B, 40]
```

---

## 2. 🧠 **Model Architecture**

### We use:

* **BERT-base** (12 layers, 768 hidden dim, 12 heads)
* **Reward head**: 1-layer MLP on final token’s hidden state (typically `[CLS]` token)

---

### Forward Pass: Preferred Input

```python
outputs = BERT(input_ids, attention_mask, token_type_ids)
```

#### BERT Output:

```python
last_hidden_state: [B, L, H] = [B, 40, 768]
```

#### Extract \[CLS] token:

```python
cls_embedding: [B, H] = last_hidden_state[:, 0, :]  # [B, 768]
```

#### Reward head (linear projection):

```python
reward: [B, 1] = Linear(768 → 1)(cls_embedding)
```

---

### Repeat for dispreferred input:

* Compute $r^+$ and $r^-$ → both shapes: `[B, 1]`
* Squeeze to shape `[B]` for loss computation.

---

## 3. 📉 Loss Function

We use the **pairwise logistic loss**:

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \sigma(r^+_i - r^-_i)
$$

```python
r_diff = r_preferred - r_dispreferred       # shape: [B]
loss = -torch.log(torch.sigmoid(r_diff))    # shape: [B]
loss = loss.mean()                          # scalar
```

This encourages $r^+ > r^-$.

---

## 4. 🔄 Training Procedure

### Step-by-step:

1. Collect human preferences: tuples of (prompt, preferred, dispreferred)
2. Tokenize both pairs using BERT tokenizer
3. Pad to same length
4. Encode both via BERT → get CLS embeddings → project to scalar
5. Compute difference of scores
6. Apply sigmoid loss
7. Backpropagate and update parameters

> You can optionally **freeze BERT** and train only the reward head if data is limited.

---

## 5. 🧪 Evaluation Metrics

* Accuracy: % where $r^+ > r^-$
* AUC: if you treat preferences as binary classification
* Spearman/Pearson correlation (if comparing to real-valued ratings)

---

## 🧱 Summary of Tensors

| Component           | Tensor Shape     | Notes                            |
| ------------------- | ---------------- | -------------------------------- |
| `input_ids`         | `[B, L]`         | tokenized prompt + response      |
| `attention_mask`    | `[B, L]`         | 1 for real tokens, 0 for padding |
| `token_type_ids`    | `[B, L]`         | 0 for prompt, 1 for completion   |
| `last_hidden_state` | `[B, L, 768]`    | BERT output for each token       |
| `cls_embedding`     | `[B, 768]`       | first token representation       |
| `reward`            | `[B, 1]` → `[B]` | scalar reward per input          |
| `loss`              | `scalar`         | pairwise logistic loss           |

---

## 📦 Code Skeleton (PyTorch)

```python
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.reward_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        reward = self.reward_head(cls_embed).squeeze(-1)  # [B]
        return reward
```

---

## 📌 Notes:

* For long completions, use **RoBERTa** or **Longformer** (BERT is capped at 512 tokens).
* For image-text or multimodal reward models (e.g., in Flamingo or GPT-4V), visual tokens are fused at the input stage.

---

## Block diagram of the RLHF finetuning
Of course. Here is a thorough block diagram and explanation of the Reinforcement Learning from Human Feedback (RLHF) fine-tuning process, broken down into its core stages.

The overall goal of **RLHF** is to align a pre-trained language model with human preferences, making it more helpful, harmless, and honest. This is achieved not by direct programming, but by training the model on a "reward" signal derived from what humans judge to be good responses.

The process primarily consists of two major phases after an initial Supervised Fine-Tuning (SFT) step:
1.  **Training the Reward Model**: Learning what humans prefer.
2.  **Fine-tuning the LLM with Reinforcement Learning**: Optimizing the LLM to generate responses that score high on that preference model.

---

### Phase 1: Training the Reward Model (RM)

Before we can use RL, we need a way to automatically score the LLM's outputs. This is the job of the **Reward Model**. It's a separate LLM trained to take a prompt and a generated completion and output a scalar score representing how much a human would prefer that completion.

**Process:**
1.  **Generate Data**: A prompt is selected, and the initial LLM generates several different completions.
2.  **Human Ranking**: A human labeler is shown these completions and ranks them from best to worst.
3.  **Train RM**: This preference data (e.g., for a given prompt, completion A is better than B, C, and D) is used to train the Reward Model. The model learns to assign a higher reward score to the completions humans preferred.

**Block Diagram: Reward Model Training**
```
┌──────────────────┐
│  Prompt Dataset  │
└────────┬─────────┘
         │ (Feedforward: Sample Prompt x)
         ▼
┌──────────────────┐
│       SFT LLM    │
└────────┬─────────┘
         │ (Feedforward: Generate multiple completions y1, y2, y3...)
         ▼
┌──────────────────┐
│  Human Labeler   │
└────────┬─────────┘
         │ (Feedback: Rank completions, e.g., y2 > y1 > y3)
         ▼
┌─────────────────────────────────┐
│ Train Reward Model (RM)         │
│ (Supervised Learning on Pairs)  │
└────────┬────────────────────────┘
         │ (Output: A trained RM that predicts human preference)
         ▼
┌──────────────────┐
│   Reward Model   │
└──────────────────┘
```

---

### Phase 2: Fine-tuning the LLM with RL

This is the core RLHF loop. Here, the pre-trained LLM acts as the **policy**, the action is generating a completion, and the **reward** comes from the Reward Model we just trained. The goal is to update the LLM's weights (its policy) to maximize this reward.

**Process:**
1.  **Prompt**: A new prompt is sampled from the dataset.
2.  **Generate**: The current LLM (the policy) generates a completion for the prompt.
3.  **Reward**: The prompt-completion pair is passed to the frozen **Reward Model**, which returns a scalar reward score.
4.  **Update**: The **RL Algorithm** (commonly PPO - Proximal Policy Optimization) uses this reward to calculate a policy gradient and update the LLM's weights. A key part of this step is a **KL-divergence penalty**. This penalty measures how much the LLM has strayed from its original SFT version and prevents it from over-optimizing for the reward signal to the point of generating nonsensical text (a problem known as "reward hacking").

**Block Diagram: RLHF Fine-tuning Loop**
```
                                        ┌─────────────────────────┐
                               ┌────────│  Reference SFT Model    │
                               │        │  (Frozen, π_ref)        │
                               │        └───────────┬─────────────┘
(Feedforward: Sample Prompt x) │                    │ (Calculates P(y|x) for penalty)
                               │                    │
┌──────────────────┐           │                    ▼
│  Prompt Dataset  ├───────────►┌-──────────────────┐      ┌──────────────────────────┐
└──────────────────┘           │      LLM Policy    ├─────►│ RL Algorithm (e.g., PPO) │
                               │       (π_θ) LORA   │      └───────────┬──────────────┘
                               └─────────┬─────────-┘                  │ (Feedback: Update Gradients ∇θ)
                                         │                            │
                     (Feedforward: Generate Completion y)             │
                                         │                            │
                                         ▼                            │
                               ┌───────────────────┐                  │
                               │   Reward Model    │                  │
                               │      (RM)         ├──────────────────┘
                               │(Frozen, from Ph.1)│  (Feedback: Reward Signal r)
                               └───────────────────┘

```
This entire second phase is an iterative loop that continuously refines the LLM's ability to produce outputs that align with the human preferences captured by the Reward Model.


# PPO RL Algorithm

Of course. Let's break down the **Proximal Policy Optimization (PPO)** algorithm from the ground up. I'll build from the core intuitions to the technical details of the architecture and the loss function, just as you'd need for a thorough tutorial.

### Part 1: The Intuition - Stable Steps in the Right Direction

Imagine you're teaching a robot to walk. In early reinforcement learning algorithms (like standard Policy Gradient methods), it was like giving the robot a huge push in what you *thought* was the right direction after each successful step. Sometimes this worked, but other times you'd push it so hard it would completely lose its balance and fall over, ruining all the progress it had made. The learning steps were too big and unstable.

**The Core Problem PPO Solves:** How can we ensure the robot takes confident steps toward learning, without letting any single step be so large that it destabilizes the entire learning process?

PPO's answer is simple but powerful: **Take small, conservative steps.** It improves the policy (the robot's walking strategy) in a way that is "proximal" (i.e., close) to the previous version of the policy. It avoids making drastic changes, which leads to much more stable and reliable training.

It's a successor to an algorithm called **Trust Region Policy Optimization (TRPO)**, which solved the same problem but was much more mathematically complex and computationally expensive. PPO provides a simpler way to achieve the same goal of taking controlled, stable steps.

---

### Part 2: The Model Architecture - The Actor and the Critic

PPO uses a very common and effective architecture in modern reinforcement learning called the **Actor-Critic** model. Think of this as a team of two neural networks working together to learn a task.

1.  **The Actor (The Policy):** This network is the "doer." It looks at the current state of the environment (e.g., the robot's sensor readings) and decides on an action to take (e.g., which motors to move). The Actor *is* the policy; its goal is to learn the best possible action for any given state.

2.  **The Critic (The Value Function):** This network is the "evaluator." It also looks at the current state, but instead of choosing an action, it estimates how "good" that state is. It outputs a single number called the **value**, which represents the total future reward we can expect to get from this state. Its job is to "criticize" the states the Actor ends up in.

**How They Work Together:**
The Actor takes an action. The environment gives a reward and a new state. The Critic then looks at that new state and says, "Hmm, based on my calculations, this new state is worth X future reward." This information is used to judge the Actor's last move.

This leads us to a crucial concept: the **Advantage Function**.

#### The Advantage Function ($A(s, a)$)

Instead of just looking at the raw reward, PPO wants to know: "Was the action we took *better or worse* than the average action we could have taken from that state?" This is what the Advantage Function tells us.

A simple way to think about it is:

$A(s, a)$ = (Reward we actually got) - (The Critic's estimate of the state's value)

* If **Advantage is positive**: The action was better than expected! We should increase the probability of taking this action in the future.
* If **Advantage is negative**: The action was worse than expected. We should decrease the probability of taking this action in the future.

This "advantage" signal is a much more stable and effective learning signal than just using raw rewards.

**Model Architecture Diagram:**

```
                                      +-----------------+
                                      |   Environment   |
                                      +--------+--------+
                                               |
                          (Action a_t)         | (State s_t, Reward r_t)
                                               |
                      +------------------------+------------------------+
                      |                                                 |
                      |                                                 |
                      v                                                 v
              +--------------+                                  +--------------+
              | Actor (Policy) | -- (looks at state s_t) -------> | Critic (Value)|
              |   π_θ(a|s)     |                                  |   V_φ(s)     |
              +--------------+                                  +--------------+
                      |                                                 |
(Decides on Action)   |                                                 | (Estimates Value)
                      |                                                 |
                      |                                                 |
                      +------------------------+------------------------+
                                               |
                                               v
                                   +-------------------------+
                                   |      PPO Algorithm      |
                                   | (Calculates Advantage & |
                                   |    Updates Actor & Critic) |
                                   +-------------------------+
```

---

### Part 3: The PPO Loss Function - The "Secret Sauce"

This is where PPO's core innovation lies. The total loss is a combination of three different components.

#### 1. The Policy Loss (The Clipped Surrogate Objective)

This is the main event. The goal is to make actions with a positive advantage more likely, and actions with a negative advantage less likely.

Let's define a ratio:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

* $\pi_\theta(a_t | s_t)$ is the probability of taking action $a_t$ in state $s_t$ with the **current** policy.
* $\pi_{\theta_{old}}(a_t | s_t)$ is the probability from the **old** policy (before this round of updates).

If $r_t(\theta) > 1$, the new policy is more likely to take that action. If $r_t(\theta) < 1$, it's less likely.

A naive objective would be to just multiply this ratio by the advantage: $r_t(\theta) \hat{A}_t$. But this can lead to those huge, unstable updates we want to avoid.

PPO introduces a **clipping mechanism**:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

Let's break down that `min` function:

* **First term:** $r_t(\theta) \hat{A}_t$ is our normal objective.
* **Second term:** `clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t` is the "clipped" version. The ratio $r_t(\theta)$ is not allowed to go outside the range $[1 - \epsilon, 1 + \epsilon]$ (where $\epsilon$ is a small number, usually 0.2).

**The Intuition of the `min` function:**

* **Case 1: Advantage $\hat{A}_t$ is positive.** (The action was good).
    The loss function encourages us to increase $r_t(\theta)$, making the action more likely. But the `clip` function puts a ceiling on this. Once the policy moves far enough away from the old one (i.e., $r_t(\theta) > 1 + \epsilon$), the objective function flattens out. This prevents us from getting too greedy and making the update too large. We take the minimum of the normal objective and the clipped one, so we are being pessimistic and taking the smaller step.

* **Case 2: Advantage $\hat{A}_t$ is negative.** (The action was bad).
    The loss function encourages us to decrease $r_t(\theta)$. The `clip` function puts a floor on this at $1 - \epsilon$. This prevents us from overreacting and making a good action extremely unlikely just because of one bad outcome.

This clipping is the simple, elegant solution that keeps the policy updates small and stable.

#### 2. The Value Function Loss

This part is much simpler. It's the loss for the Critic network. We want the Critic's value estimate $V_\phi(s_t)$ to be as close as possible to the actual returns we observed ($V_t^{\text{target}}$). This is a standard mean squared error loss.

$$L^{VF}(\phi) = \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - V_t^{\text{target}})^2 \right]$$

This loss trains the Critic to become a more accurate estimator of how good each state is.

#### 3. The Entropy Bonus

Entropy is a measure of randomness or unpredictability. In this context, we want to encourage the policy to be a little random. Why? To promote **exploration**. If the policy becomes too certain about its actions too early, it might get stuck in a suboptimal strategy and never explore better options.

By adding an entropy term to the loss, we give the agent a small reward for being uncertain. This encourages it to keep trying new things, which can lead to finding better long-term solutions.

$$L^S(\theta) = \hat{\mathbb{E}}_t \left[ S[\pi_\theta](s_t) \right]$$

Where $S$ is the entropy of the policy.

### The Final Loss Function

The total loss for PPO is a weighted sum of these three components:

$$L(\theta, \phi) = L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 L^S(\theta)$$

* $c_1$ and $c_2$ are coefficients that balance the importance of the value loss and the entropy bonus.
* We subtract the value loss because we are typically performing gradient *ascent* on the policy objective (making it bigger) but gradient *descent* on the value error (making it smaller). Including it this way allows a single optimizer to handle everything.

By optimizing this combined objective, PPO ensures that the Actor learns to take better actions stably, the Critic learns to predict outcomes accurately, and the whole system maintains a healthy level of exploration.

# Reward Hacking
# Constitutional AI
# RLAIF
# Model Optimization
## Prompt engineering vs prompt tuning(PEFT technique) vs fine-tuning(PEFT or non peft)

# Problems with LLMs:
  1-  **Hallucination**
  2- Knowledge cut-off issue e.g., the president of the USA is Barack Obama (at the time of training, but someone else now)
# Connecting LLMS to output sources
  1- **RAG Retrieval augmented Generation**
  -  Help to update the LLM knowledg
  2- **Chain-of-thoughts prompting**
  3- **Program Aided Language Models (PAL)**
  4- **ReAct: Combining Reasoning and Action**
## LangChain
