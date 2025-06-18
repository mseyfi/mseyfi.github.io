[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

<br>


## [![ControlNet](https://img.shields.io/badge/ControlNet-Adding_Conditional_Control_to_Text_to_Image_Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/ControlNet)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
ControlNet is an advanced extension of diffusion models that introduces additional control mechanisms, allowing for precise guidance over the generation process. By integrating control signals (e.g., edge maps, segmentation masks, poses), ControlNet enables the generation of images that adhere to specific structural or semantic constraints provided by the user.
<p></p>
</div>


## [![CFG](https://img.shields.io/badge/CFD-Classifier_Free_Diffusion-blue?style=for-the-badge&logo=github)](../posts/CFG)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Classifier-Free Diffusion is a powerful technique in generative modeling, particularly within diffusion models, that enhances the quality and controllability of generated outputs without relying on an external classifier. This comprehensive guide will delve into the intricacies of classifier-free diffusion, covering its principles, training and inference processes, intuitive explanations, and practical implementations in tasks like image inpainting, super-resolution, and text-to-image generation.
<p></p>
</div>


## [![Classifier-Guided Diffusion](https://img.shields.io/badge/CGD-Classifier_Guided_Diffusion-blue?style=for-the-badge&logo=github)](../posts/Classifier_Guided_Diffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Classifier-guided diffusion is a powerful technique in generative modeling that leverages an external classifier to steer the generation process toward desired attributes or classes. This method enhances the quality and controllability of generated data, such as images, by integrating class-specific information during the diffusion process.
<p></p>
</div>


## [![Latent Diffusion](https://img.shields.io/badge/LDM-Latent_Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/StableDiffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Stable Diffusion is a powerful generative model that synthesizes high-quality images guided by textual/another modality descriptions. It leverages the strengths of Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs) to produce images efficiently and effectively.
 <p></p>
</div>

## [![NCSN](https://img.shields.io/badge/NCSN-Noise_Conditional_Score_Networks/Score_Based_Generative_Models-blue?style=for-the-badge&logo=github)](../posts/NCSN)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
A <b> Score-based Generative Model </b>  learns the gradient of the log-probability (the “score”) for a family of <b> noisy </b>  versions of data. Instead of directly learning a generative model $ p(x) $, we train a network $ s_\theta(x, \sigma) $ that approximates:

$$
\nabla_x \log p_\sigma(x) 
$$

$\text{where} \quad p_\sigma(x)$ is the distribution of data  <b> corrupted </b>  by noise of scale $\sigma$. Once we learn a good approximation of the score $\nabla_x \log p_\sigma(x)$, we can sample from the (clean) distribution by <b> progressively denoising </b>  data using <b> Langevin dynamics </b>  (or an equivalent Stochastic Differential Equation).
<p></p>
</div>


## [![Diffusion Models DDPM](https://img.shields.io/badge/DDPM-Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/Diffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
In this guide, we'll provide sample code for training and inference of a diffusion model, specifically focusing on a Denoising Diffusion Probabilistic Model (DDPM). We'll define the structure for the encoder and decoder using a simplified UNet architecture. Each line of code includes inline comments explaining its purpose, along with the tensor shapes.
 <p></p>
</div>


## [![GLIP](https://img.shields.io/badge/GLIP-Grounded_Language_Image_Pre_training-blue?style=for-the-badge&logo=github)](../posts/GLIP)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
GLIP (Grounded Language-Image Pre-training) is a unified model architecture that bridges the gap between vision and language by integrating object detection and phrase grounding tasks. It leverages both visual and textual data to perform object detection conditioned on textual descriptions, enabling the model to recognize objects based on their semantic meanings.
<p></p>
</div>


## [![CLIP](https://img.shields.io/badge/CLIP-Learning_Transferable_Visual_Models_From_Natural_Language_Supervision-blue?style=for-the-badge&logo=github)](../posts/CLIP)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Learning Transferable Visual Models From Natural Language Supervision" is a groundbreaking paper by OpenAI that introduces CLIP (Contrastive Language-Image Pre-training). CLIP learns visual concepts from natural language supervision by jointly training an image encoder and a text encoder to predict the correct pairings of images and texts.
<p></p>
</div>


## [![texttoimage](https://img.shields.io/badge/Text_to_Image-grey?style=for-the-badge&logo=github)](../posts/TextToImage)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
notable text-to-image generation models along with their corresponding research papers, sorted by the year they were published:
 <p></p>
</div>

## [![LORA](https://img.shields.io/badge/LORA-Low_Rank_Adaptation-blue?style=for-the-badge&logo=github)](../posts/LLM/LORA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large-scale pre-trained models. It allows us to adapt a model by introducing low-rank trainable matrices into certain parts of the network while keeping the original pre-trained weights frozen. <p></p>
</div>

## [![EVALUATION](https://img.shields.io/badge/Evaluation-Evaluation_Metrics_for_LLMs-blue?style=for-the-badge&logo=github)](../posts/LLM/EvaluationMetricsLLM)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Language models are evaluated across diverse tasks such as next-token prediction, text classification, summarization, translation, code generation, and question answering. Each task requires a suitable metric that reflects model performance both quantitatively and qualitatively.
</div>

## [![Taks](https://img.shields.io/badge/LLM_TASKS-A_Comprehensive_Guide_to_LLM_Tasks:_From_Fine_Tuning_to_Advanced_Applications-blue?style=for-the-badge&logo=github)](../posts/LLM/LLM-Tasks)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
A single, powerful pre-trained Large Language Model is a versatile foundation that can be adapted—or "fine-tuned"—to excel at a wide array of specific tasks. The key to this versatility lies in how we frame the task and format the data.

This tutorial provides a detailed breakdown of the most common and important tasks an LLM can perform, explaining for each: its goal, the data format for fine-tuning, a practical example, and the underlying mechanics.</div>

## [![Spec](https://img.shields.io/badge/Speculative_Decoding-Speculative_Decoding_For_Fast_Inference-blue?style=for-the-badge&logo=github)](../posts/LLM/Speculative-Decoding)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Even with highly optimized systems, the process of generating text from Large Language Models (LLMs) faces a final, stubborn bottleneck: <b>sequential decoding</b>. Generating one token requires a full, time-consuming forward pass through the model. Because this must be done one token at a time, the overall speed is limited by memory latency.
</div>

## [![FlashAttention](https://img.shields.io/badge/FLASHATTENTION-Fast_and_Memory_Efficient_Exact_Attention_with_IO_Awareness-blue?style=for-the-badge&logo=github)](../posts/LLM/FlashAttention)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
FlashAttention is a pivotal innovation that addresses one of the most fundamental bottlenecks in the Transformer architecture. Unlike architectural changes like MoE or GQA, FlashAttention is a groundbreaking implementation that doesn't change the mathematics of attention but radically alters how it's computed on the hardware.
</div>

## [![ALBERT](https://img.shields.io/badge/ALBERT-Cross_Layer_Parameter_Sharing:_Efficient_FFN_Structures-blue?style=for-the-badge&logo=github)](../posts/LLM/ALBERT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While innovations like MoE and SwiGLU focus on redesigning the FFN block within a single Transformer layer, Cross-Layer Parameter Sharing tackles efficiency from a different angle: it re-evaluates the relationship between the layers themselves. It asks a simple but profound question: "Does every layer in a deep network truly need its own unique set of weights?"
</div>

## [![MOE](https://img.shields.io/badge/MOE_Mixture_of_Experts:_FFN-Efficient_FFN_Structures-blue?style=for-the-badge&logo=github)](../posts/LLM/Efficient-FFN)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While attention mechanisms get much of the spotlight, the Feed-Forward Network (FFN) is the computational workhorse of the Transformer. Typically comprising two-thirds of a model's parameters, the FFN block is a critical target for efficiency innovations. As of mid-2025, the strategies have evolved far beyond simply tweaking dimensions.
</div>

## [![SwiGlu](https://img.shields.io/badge/SWiGLU-The_Modern_FFN_for_High_Performance_Transformers-blue?style=for-the-badge&logo=github)](../posts/LLM/SWiGLU)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
While Mixture of Experts (MoE) offers a path to scale models to enormous parameter counts via sparsity, another critical innovation has optimized the performance of the core "dense" Transformer block itself. This is the <b>Gated Linear Unit (GLU)</b>, and specifically its most successful variant, <b>SwiGLU</b>.

As of mid-2025, SwiGLU has replaced the traditional FFN in most state-of-the-art dense language models (like Meta's Llama series, Google's PaLM, and Mistral's dense models). This tutorial explores the intuition, mathematics, and implementation behind this powerful and efficient architecture.
</div>


## [![KV](https://img.shields.io/badge/KV_Caching-How_to_optimize_the_inference_by_caching_key_value-blue?style=for-the-badge&logo=github)](../posts/LLM/KV-Caching)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Generative Large Language Models (LLMs) have transformed technology, but their power comes at a cost: inference (the process of generating new text) can be slow. The primary bottleneck is the "autoregressive" way they produce text—one token at a time. A simple but profound optimization called the KV Cache is the key to making this process practical and fast.

This tutorial will give you a deep understanding of what the KV Cache is, the critical reason why we only cache Keys (K) and Values (V) but not Queries (Q), and how to implement it in code.
</div>

## [![MQA](https://img.shields.io/badge/MQA_GQA-Multi_Query_Attention,_Grouped_Query_Attention-blue?style=for-the-badge&logo=github)](../posts/LLM/MQA-GQA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
The Transformer architecture's self-attention mechanism is the engine of modern AI. However, as models and their context windows grow, the computational and memory costs of standard Multi-Head Attention (MHA) become a significant bottleneck, especially during inference.

This tutorial provides an in-depth exploration of two powerful solutions: Multi-Query Attention <b>(MQA)</b> and Grouped-Query Attention <b>(GQA)</b>. Understanding these architectural details, including their mathematical foundations and impact on memory, is essential for grasping how models like Google's Gemini, Meta's Llama 3, and Mistral AI's models operate so efficiently.
</div>

## [![Linformer](https://img.shields.io/badge/Linformer-A_Primary_Example_of_Explicit_Low_Rank_Attention-blue?style=for-the-badge&logo=github)](../posts/LLM/Linformer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Overview of Linformer, why we need it, and side-by-side pseudo-code comparing traditional self-attention to Linformer self-attention. We’ll keep the example to single-head attention for clarity, but in practice you would typically use multi-head attention (with separate projections for each head).

</div>

## [![Performer](https://img.shields.io/badge/Performer-Linear_Kernel_approximation_of_Attention-blue?style=for-the-badge&logo=github)](../posts/LLM/Performer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Performers are Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attentionkernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. Performer demonstrates competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.
</div>

## [![Span-Corruption](https://img.shields.io/badge/Span_Corruption-Span_Corruption_In_Encoder_Decoder_LLM_Architectures(T5)-blue?style=for-the-badge&logo=github)](../posts/LLM/SpanCorruption)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Span corruption is a key self-supervised pre-training objective introduced in the T5 (Text-to-Text Transfer Transformer) model. It's designed to make the model learn to reconstruct missing contiguous spans of text, which is crucial for tasks like denoising, summarization, and machine translation.
</div>


## [![Encoder-Decoder](https://img.shields.io/badge/Encoder_Decoder-Encoder_Decoder_Transformers(T5)-blue?style=for-the-badge&logo=github)](../posts/LLM/Encoder-Decoder)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Encoder-decoder models are the canonical choice for any task that maps one sequence of arbitrary length to another. They form a powerful and flexible framework for problems where the input and output have different lengths, structures, or even languages.
</div>

## [![Decoder](https://img.shields.io/badge/Decoder_Only-Decoder_Only_Transformers_(Llama,_GPT)-blue?style=for-the-badge&logo=github)](../posts/LLM/Decoder-Only)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
The core task of a decoder-only model is autoregressive text generation. The model learns to predict the very next word given a sequence of preceding words. It generates text one token at a time, feeding its own previous output back in as input to predict the next token. This simple, self-supervised objective, when applied at a massive scale, enables the model to learn grammar, facts, reasoning abilities, and style.
</div>

## [![RoPE](https://img.shields.io/badge/ROPE-Rotary_Positional_Embedding-blue?style=for-the-badge&logo=github)](../posts/LLM/ROPE)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
ROPE injects positional information into queries and keys by rotating their components in complex space. Instead of adding positional encodings (like in vanilla Transformers), ROPE rotates the vector based on position — and this rotation preserves distance and ordering relationships.
</div>

## [![RoBERTa](https://img.shields.io/badge/RoBERTa-A_Robustly_Optimized_BERT_Pretraining_Approach-blue?style=for-the-badge&logo=github)](../posts/LLM/Roberta)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
After BERT established the paradigm of pre-training and fine-tuning, the natural next step in the scientific process was to ask: "Was this done optimally?" The original BERT paper left several questions unanswered regarding its design choices. Was the Next Sentence Prediction task truly necessary? How much did the data size and other hyperparameters matter?

This brings us to our next topic: <b> RoBERTa </b> , a 2019 model from Facebook AI that stands for <b>R</b>obustly <b>O</b>ptimized <b>BERT A</b>pproach. RoBERTa is not a new architecture. Rather, it is a meticulous study that takes the original BERT architecture and systematically evaluates its pre-training recipe, resulting in a significantly more powerful model.

Think of BERT as the revolutionary prototype. RoBERTa is the production model, fine-tuned and optimized for maximum performance. Let's begin the tutorial.
</div>

## [![Encoder](https://img.shields.io/badge/Encoder_Only-Encoder_Only_Transformers(BERT)-blue?style=for-the-badge&logo=github)](../posts/LLM/BERT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
BERT (Bidirectional Encoder Representations from Transformers), developed by Google AI in 2018, marked a paradigm shift in natural language processing by introducing deep bidirectional context modeling through the Transformer architecture. Unlike traditional left-to-right or right-to-left language models, BERT reads entire sequences bidirectionally, enabling better understanding of linguistic context.
</div>

