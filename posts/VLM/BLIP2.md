## [![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](/)
## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computer_Vision-green?style=for-the-badge&logo=github)](/main_page/CV)
## [![CV](https://img.shields.io/badge/VLMs-Selected_Topics_in_Vision_Language_Models-orange?style=for-the-badge&logo=github)](VLMs)

## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

### The Intuition: Why We Need BLIP-2

The world of AI in the early 2020s was marked by a significant trend: the meteoric rise of **Large Language Models (LLMs)** like GPT-3, PaLM, and LLaMA. These models, trained on vast text corpora, demonstrated astonishing capabilities in understanding, reasoning, and generating human-like text. Concurrently, vision-language (VL) models like the original BLIP, while powerful, faced a critical challenge: **prohibitive training costs**.

The standard paradigm for VL models was end-to-end pre-training of the entire architecture—both the vision encoder and the language model components. As state-of-the-art LLMs grew to hundreds of billions of parameters, creating a vision-language model of a similar scale by training it from scratch became computationally and financially infeasible for most researchers. This created a bottleneck for progress.

This is the problem BLIP-2 was designed to solve. The core intuition is both simple and profound:

![im1](/images/BLIP2-Fig1.png)

*Fig.1  Overview of BLIP-2’s framework. We pre-train a
lightweight Querying Transformer following a two-stage strategy to bridge the modality gap. The first stage bootstraps visionlanguage representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen LLM, which enables zero-shot instructed image-totext generation (see Figure 4 for more examples).*

**Instead of building a massive, monolithic vision-language model from scratch, can we leverage the power of existing, pre-trained, *frozen* models and teach them to work together?**

BLIP-2 proposes to act as a "smart translator" or "bridge" between a powerful, off-the-shelf **frozen image encoder** and a powerful, off-the-shelf **frozen LLM**. The key insight is that the most expensive components (the vision backbone and the LLM) already contain immense knowledge. The challenge is not to re-learn this knowledge, but to bridge the "modality gap" between them. By training only a small, lightweight bridging module, BLIP-2 drastically reduces the number of trainable parameters and the overall pre-training cost, making state-of-the-art vision-language modeling more accessible and efficient.

This "bridging" component is the heart of BLIP-2: the **Querying Transformer (Q-Former)**.

### The BLIP-2 Model Structure: A Symphony of Three Components

The BLIP-2 architecture is a masterclass in modularity, composed of three main parts:

1. **A Frozen Image Encoder:** This is a standard, pre-trained vision model whose job is to extract visual features from an input image. The original work uses a Vision Transformer (ViT-L/14 from CLIP), but any powerful vision model could work. Crucially, its weights are **frozen** during the entire pre-training process. It acts solely as a feature provider.

2. **A Frozen Large Language Model (LLM):** This provides the language understanding and generation capabilities. BLIP-2 was tested with decoder-based LLMs like OPT and encoder-decoder LLMs like Flan-T5. Like the image encoder, the LLM's weights are also **frozen**. It brings its vast linguistic knowledge to the table without needing to be retrained.

3. **The Querying Transformer (Q-Former):** This is the lightweight, trainable module that connects the two frozen giants. The Q-Former is a cleverly designed transformer model tasked with a single, critical job: to extract visual features from the frozen image encoder that are most relevant to the text. It acts as an information bottleneck, distilling the rich, high-dimensional visual information into a few concise feature vectors that the LLM can understand.

   Let's look closer at the Q-Former. It contains two main sub-modules that share the same self-attention layers:

   - **Learnable Query Embeddings:** The Q-Former introduces a fixed number (e.g., 32) of learnable embedding vectors. These queries are persistent and are not tied to any specific input. Think of them as a set of "questions" the model learns to ask about the image.
   - **Image Transformer:** This module interacts with the frozen image encoder. Its learnable queries interact with the image patch embeddings via **cross-attention** to extract visual features.
   - **Text Transformer:** This is a standard BERT-style text encoder-decoder. It can function as both a text encoder (understanding) and a text decoder (generation).

   The brilliance of the Q-Former lies in its attention mechanisms. The learnable queries can attend to the image patches (via cross-attention), they can attend to each other (via self-attention), and they can attend to input text (via the same cross-attention layers). This allows them to become a flexible representation of the visual scene, conditioned by text when available.

Let's clarify: **The Q-Former is a single, unified transformer.**

It is not a Siamese network with two parallel streams. Instead, think of it as a single transformer that processes a *concatenated sequence* of two different types of tokens: the learnable **queries** and the input **text tokens**. The "submodule" concept refers to the different operations that are applied to these two types of tokens as they pass through the single transformer architecture.

*   The **"image transformer" submodule** refers to the operations involving the learnable queries and their interaction with the frozen image encoder's features via cross-attention.
*   The **"text transformer" submodule** refers to the operations involving the text tokens.

Because they are processed as one long sequence in the self-attention layers, they can interact. The degree and direction of this interaction are precisely controlled by the attention masks.

### Mathematical and Algorithmic Explanation

Let's define the inputs and their dimensions, following the paper's examples.

*   **Frozen Image Features ($I$):** The output of the frozen image encoder (e.g., ViT-L/14).
    *   $I \in \mathbb{R}^{B \times N_p \times D_\text{img}}$
    *   $B$ = Batch size
    *   $N_p$ = Number of image patches (e.g., 257)
    *   $D_\text{img}$ = Dimension of image features (e.g., 1024)

*   **Learnable Queries ($Q_\text{learn}$):** A set of trainable embeddings that are part of the Q-Former's parameters.
    *   $Q_\text{learn} \in \mathbb{R}^{N_q \times D_q}$
    *   $N_q$ = Number of queries (e.g., 32)
    *   $D_q$ = Hidden dimension of the Q-Former (e.g., 768, for BERT-base)

*   **Input Text Embeddings ($T$):** The standard token embeddings of the input text.
    *   $T \in \mathbb{R}^{B \times N_t \times D_q}$
    *   $N_t$ = Number of text tokens
    *   $D_q$ = Hidden dimension (must match queries, e.g., 768)

---

![im2](/images/BLIP2-Fig2.png)

*Fig.2 (Left) Model architecture of Q-Former and BLIP-2’s first-stage vision-language representation learning objectives. We jointly
optimize three objectives which enforce the queries (a set of learnable embeddings) to extract visual representation most relevant to the
text. (Right) The self-attention masking strategy for each objective to control query-text interaction.*

#### Inside a Q-Former Block

A Q-Former block consists of a **Self-Attention** layer, a **Cross-Attention** layer (which is only applied to the queries), and a **Feed-Forward Network (FFN)**. Let $H_q$ and $H_t$ be the query and text representations entering the block.

**1. Self-Attention (The Core Interaction):**

The queries and text are concatenated into a single sequence.

$$
X = \text{concat}(H_q, H_t) \qquad\text{where}\qquad X \in \mathbb{R}^{B \times (N_q + N_t) \times D_q}
$$

This combined sequence is fed into a standard multi-head self-attention layer. The key is the **attention mask (SMS)**, which controls which tokens can attend to which other tokens.

$$
\text{SelfAttn}_text{output} = \text{MultiHeadSelfAttention}(Q=X, K=X, V=X, \text{Mask}=M)
$$

The mask $M$ changes depending on the pre-training objective (as seen in Figure 2):

*   **For ITM (bi-directional):** $M$ allows all tokens to attend to all other tokens.
*   **For ITC (unimodal):** $M$ is block-diagonal. $H_q$ can only attend to $H_q$, and $H_t$ can only attend to $H_t$.
*   **For ITG (multimodal causal):** $M$ allows $H_q$ to attend to all $H_q$. It allows $H_t$ to attend to all $H_q$ but only to *previous* tokens within $H_t$.

**2. Cross-Attention (Injecting Visual Information):**

This step **only applies to the query representations**. The output from the self-attention layer is split back into query and text representations. Let's call the query part $\text{SelfAttn}_q$.

$$
\text{CrossAttn}_q = \text{MultiHeadCrossAttention}(Q=\text{SelfAttn}_q, K=I, V=I)
$$

The text representations bypass this step entirely. This is crucial: **only the queries "see" the image.** They act as a bottleneck, summarizing the visual information. The paper notes this happens in every other transformer block.

**3. Feed-Forward Network (FFN):**

The final step in a block is a standard FFN applied to all representations. The output of cross-attention for queries and the output of self-attention for text are passed through the FFN.

### Tokens and Embeddings: The Building Blocks

  * **Image Tokens:** The frozen ViT takes an image, divides it into a grid of patches (e.g., 14x14 pixels), and linearly projects each patch into a vector. These vectors are the image tokens. Positional embeddings are added to retain spatial information.
  * **Text Tokens:** Text is processed using a standard LLM tokenizer (like BPE or WordPiece) into a sequence of sub-word tokens. These are then converted into text embeddings.
  * **Learnable Query Embeddings:** These are the key to the Q-Former. They are a set of $N$ vectors (e.g., $N=32$), each with a dimension $d$ (e.g., $d=768$). They are initialized randomly and are learned during the pre-training process. They are not input-dependent; they are model parameters. Their purpose is to act as a summary or a set of "experts" that learn to extract specific types of visual information (e.g., one query might learn to focus on objects, another on colors, another on textures).

### The Two-Stage Pre-training Strategy

BLIP-2's training is elegantly divided into two stages to first teach the Q-Former how to "see" and then how to "talk" to the LLM.

#### Stage 1: Vision-Language Representation Learning

**Goal:** Train the Q-Former to extract visual representations that are aligned with text. In this stage, the **frozen LLM is not used**. We only use the frozen Image Encoder and the Q-Former.

**Input-Output Training Pairs:** The input is a standard image-text pair `(Image, Text)` from a web dataset.

This stage uses three interconnected loss functions, computed simultaneously, to train the Q-Former.

**1. Image-Text Contrastive Loss ($$\mathcal{L}_{itc}$$):**

  * **Intuition:** To make the model understand which images and texts belong together on a high level. It aligns the visual and text representations in a shared embedding space.

  * **Process:**

    1.  The image passes through the frozen ViT to get patch embeddings.
    2.  The Q-Former's learnable queries interact with the image patches via cross-attention. The output embedding of one of the queries (which is now visually-grounded) is chosen as the visual representation, $$q_{img}$$.
    3.  The text passes through the Q-Former's text encoder to get a text representation, $$t_{text}$$.
    4.  Similarity scores are calculated between $$q_{img}$$ and $$t_{text}$$ for all pairs in a batch. The model is trained to maximize the similarity for matched pairs and minimize it for mismatched pairs.

  * **Mathematics:** For a batch of $N$ pairs, the similarity is $$s(I_i, T_j) = q_{img}(I_i)^T \cdot t_{text}(T_j)$$. The loss is a standard contrastive cross-entropy loss over these similarities, computed for both image-to-text and text-to-image directions.

$$
p^{i2t}_j = \frac{\exp(s(I_i, T_j))}{\sum_{k=1}^{N} \exp(s(I_i, T_k))}
$$

and

$$
\mathcal{L}_{itc} = - \frac{1}{2N} \sum_{i=1}^{N} (\log p^{i2t}_i + \log p^{t2i}_i)
$$

**2. Image-Text Matching Loss ($$\mathcal{L}_{itm}$$):**

  * **Intuition:** To teach the model a fine-grained understanding of whether a specific text accurately describes an image. This is a binary classification task.

  * **Process:**

    1.  The output of the visually-grounded queries (which have "seen" the image) are fed as input to the Q-Former's text transformer, along with the text embeddings.
    2.  A special `[CLS]` token is used, and its final output embedding serves as a fused representation of the image and text.
    3.  A linear classifier on top of this `[CLS]` embedding predicts a logit for `match` vs. `not-match`.

  * **Mathematics:** This is a standard binary cross-entropy loss. Hard negatives (pairs that are semantically similar but incorrect, found using the ITC loss) are used to make the task more challenging.

    $$
    \mathcal{L}_{itm} = H(y^{itm}, p^{itm})
    $$

    where $$y^{itm}$$ is the ground truth label (1 or 0) and $$p^{itm}$$ is the predicted probability.

**3. Image-grounded Text Generation Loss ($$\mathcal{L}_{itg}$$):**

  * **Intuition:** This is the most crucial loss. It forces the learnable queries to extract *all* the visual information necessary to completely reconstruct the accompanying text.

  * **Process:**

    1.  The learnable queries interact with the frozen image encoder's outputs.
    2.  These visually-grounded queries are then fed into the Q-Former's text decoder.
    3.  The text decoder must then generate the original caption, conditioned *only* on the information provided by the queries. A causal self-attention mask is used for the text tokens.

  * **Mathematics:** This is a standard auto-regressive language modeling loss (cross-entropy). The model predicts the next token in the caption given the previous tokens and the visual information distilled into the queries.

    $$
    \mathcal{L}_{itg} = - \sum_{i=1}^{|T|} \log P(T_i | T_{<i}, Q_{img})
    $$

After Stage 1, the Q-Former has learned to convert an image into a small set of "soft instruction" vectors that encapsulate the visual content.

#### Stage 2: Vision-to-Language Generative Learning

**Goal:** Connect the trained Q-Former to the frozen LLM, teaching the LLM to understand the Q-Former's output.

**Input-Output Training Pairs:** The input is still an image-text pair `(Image, Text)`.

**Process:**

1.  The image is fed through the frozen ViT.
2.  The trained Q-Former processes the ViT output, producing the 32 visually-grounded query embeddings, $$Q_{img}$$.
3.  These query embeddings $$Q_{img}$$ are then projected by a linear layer to match the embedding dimension of the frozen LLM.
4.  These projected embeddings are prepended to the input text embeddings. For example, if the input text is "A photo of", the input to the LLM becomes `[...32 query embeddings...] [embedding for "A"] [embedding for "photo"] [embedding for "of"]`.
5.  The frozen LLM then performs its standard auto-regressive text generation, aiming to complete the sequence.
6.  The loss is calculated only on the text part of the output.

**Mathematics:** The loss is a simple language modeling loss ($$\mathcal{L}_{gen}$$), identical in form to $$\mathcal{L}_{itg}$$. However, the key difference is *what is being trained*. Here, the LLM is frozen. The loss gradients flow back through the LLM and update only the Q-Former and the linear projection layer. This fine-tunes the Q-Former to produce visual representations in a "language" that the frozen LLM can naturally comprehend and use as a prompt.

$$
\mathcal{L}_{gen} = - \sum_{i=1}^{|T|} \log P_{LLM}(T_i | T_{<i}, Q_{img})
$$

-----


![im3](/images/BLIP2-Fig3.png)

*Fig.3 BLIP-2’s second-stage vision-to-language generative pre-training, which bootstraps from frozen large language models (LLMs).
(Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5). The
fully-connected layer adapts from the output dimension of the Q-Former to the input dimension of the chosen LLM.*



![im4](/images/BLIP2-Fig4.png)

*Fig.4 Selected examples of instructed zero-shot image-to-text generation using a BLIP-2 model w/ ViT-g and FlanT5XXL, where it
shows a wide range of capabilities including visual conversation, visual knowledge reasoning, visual commonsense reasoning, storytelling,
personalized image-to-text generation, etc.*



![im5](/images/BLIP2-Fig5.png)

*Fig.5 Model architecture for VQA finetuning, where the LLM receives Q-Former’s output and the question as input, then predicts answers. We also provide the question as a condition to Q-Former, such that the extracted image features are more relevant to the question.*


### Pseudo-Code for Training

```python
# --- Setup ---
frozen_image_encoder = load_frozen_vit()
frozen_llm = load_frozen_llm()
q_former = initialize_q_former() # This is trainable
# Other components like linear layers, etc.

# --- Stage 1: Representation Learning ---
for image, text in dataloader:
    image_embeds = frozen_image_encoder(image) # Shape: [B, N_patches, D_img]
    
    # Forward pass through Q-Former for all three losses
    # query_output shape: [B, N_queries, D_qformer]
    # text_output shape: [B, N_tokens, D_qformer]
    query_output, text_output, itm_logit = q_former(image_embeds, text)

    # Loss 1: ITC
    loss_itc = calculate_contrastive_loss(query_output, text_output)
    
    # Loss 2: ITM
    loss_itm = calculate_matching_loss(itm_logit, ground_truth_match_label)
    
    # Loss 3: ITG
    # This involves using Q-Former's decoder part
    loss_itg = calculate_generation_loss(query_output, text)

    # Total loss for Stage 1
    total_loss = loss_itc + loss_itm + loss_itg
    total_loss.backward()
    optimizer.step() # Updates only Q-Former parameters

# --- Stage 2: Generative Learning ---
# After Stage 1 is complete, we use the trained Q-Former
for image, text in dataloader:
    with torch.no_grad(): # Ensure image encoder is not updated
        image_embeds = frozen_image_encoder(image)

    # Get visual features from the trained Q-Former
    query_output = q_former.extract_visua\mathcal{L}_features(image_embeds)
    
    # Prepare input for the frozen LLM
    llm_input_embeds = prepare_llm_input(query_output, text)
    
    # The LLM is frozen, but we need its output for the loss
    llm_output_logits = frozen_llm(inputs_embeds=llm_input_embeds)
    
    # Loss is calculated only on the text part of the prediction
    loss_gen = calculate_llm_generation_loss(llm_output_logits, text)

    # Gradients flow back to update Q-Former and projection layers
    loss_gen.backward()
    optimizer.step()
```

### Inference and Fine-Tuning

Once pre-training is complete, BLIP-2 is a powerful zero-shot vision-language model.

**Inference:** To use the model, you provide it with an image and a text prompt.

1.  The image is processed by the frozen ViT and the trained Q-Former to produce the 32 query embeddings.
2.  These embeddings are prepended to your tokenized text prompt.
3.  The combined sequence is fed to the frozen LLM, which generates the text completion.

  * **For Image Captioning:**
    * Image: A picture of a dog playing fetch.
    * Prompt: "a photo of"
    * Generated Output: "... a golden retriever catching a frisbee in a park."

  * **For Visual Question Answering (VQA):**
    * Image: Same picture.
    * Prompt: "Question: What is the dog playing with? Answer:"
    * Generated Output: "a frisbee."

**Fine-Tuning:** The beauty of BLIP-2 is its parameter-efficient fine-tuning. For a downstream task like medical VQA, you don't need to update the ViT or the LLM. You simply continue the Stage 2 training process on your specific dataset, fine-tuning only the Q-Former to adapt its visual extraction capabilities to the nuances of your domain (e.g., learning to recognize X-ray features). This makes adapting BLIP-2 to new tasks incredibly efficient.

### Reference

The work described here is based on the original research paper:

**Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with a Frozen Image Encoder and a Frozen Large Language Model*. In Proceedings of the 40th International Conference on Machine Learning (ICML).**
