## [![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)


<br>
<br>

## [![AnomalyDA](https://img.shields.io/badge/Anomaly-DA-g?style=for-the-badge&logo=github)](../posts/System Design/Domain_AD)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Vision-Language Models (VLMs) represent a frontier in artificial intelligence, creating systems that can see and reason about the visual world in tandem with human language. The field is incredibly diverse, with models specializing in distinct but related tasks. This guide provides a structured overview of the most prominent VLMs, categorized by their primary function.
<p></p>
</div>

## [![VLM](https://img.shields.io/badge/VLM-Vision_Language_Models-g?style=for-the-badge&logo=github)](../posts/VLM/VLMs)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Vision-Language Models (VLMs) represent a frontier in artificial intelligence, creating systems that can see and reason about the visual world in tandem with human language. The field is incredibly diverse, with models specializing in distinct but related tasks. This guide provides a structured overview of the most prominent VLMs, categorized by their primary function.
<p></p>
</div>


## [![Efficiency](https://img.shields.io/badge/Efficient_Transformers-Efficient_Techniques_in_Transformers-blue?style=for-the-badge&logo=github)](../posts/EfficientTransformers)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Vision Transformers (ViTs) have become a popular choice for image recognition and related tasks, but they can be computationally expensive and memory-heavy. Below is a list of common (and often complementary) techniques to optimize Transformers—including ViTs—for more efficient training and inference. Alongside each category, I’ve mentioned some influential or representative papers.

</div>

## [![AxialAttention](https://img.shields.io/badge/Axial_Attention-Attentions_across_axes-blue?style=for-the-badge&logo=github)](../posts/AxialAttention)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Key ideas:
  
1. Perform attention **across rows** (the width dimension, $W$) for each of the $H$ rows, independently.  
2. Then perform attention **across columns** (the height dimension, $H$) for each of the $W$ columns, independently.  
3. Each step is effectively 1D self-attention, so the cost scales like $O(H \cdot W^2 + W \cdot H^2)$ instead of $O(H^2 W^2)$.
 <p></p>

</div>


## [![TrackFormer](https://img.shields.io/badge/TrackFormer-Multi_Object_Tracking_with_Transformer-blue?style=for-the-badge&logo=github)](../posts/TrackFormer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Traditional multi-object tracking (MOT) systems often follow a two-step pipeline:
  
Detect objects in each frame independently.
Associate detections across frames to form trajectories.
This separation can lead to suboptimal solutions since detection and association are treated as separate problems. TrackFormer merges these steps by extending a Transformer-based detection architecture (inspired by DETR) to simultaneously detect and track objects. It does this by introducing track queries that carry information about previously tracked objects forward in time, allowing the network to reason about detection and association in a unified end-to-end manner. <p></p>
</div>

## [![DETR](https://img.shields.io/badge/DETR-Detection_Transformer-blue?style=for-the-badge&logo=github)](../posts/DETR)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
The Detection Transformer (DETR) is a novel approach to object detection that leverages Transformers, which were originally designed for sequence-to-sequence tasks like machine translation. Introduced by Carion et al. in 2020, DETR simplifies the object detection pipeline by eliminating the need for hand-crafted components like anchor generation and non-maximum suppression (NMS).
 <p></p>
</div>

## [![VIT](https://img.shields.io/badge/VIT-Vision_Transformers-blue?style=for-the-badge&logo=github)](../posts/VIT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Vision Transformers (ViTs) apply the Transformer architecture, originally designed for natural language processing (NLP), to computer vision tasks like image classification. ViTs treat an image as a sequence of patches (akin to words in a sentence) and process them using Transformer encoders. <p></p>
</div>

## [![VAE](https://img.shields.io/badge/VAEs-Variational_Auto_Encoders-blue?style=for-the-badge&logo=github)](../posts/VAE)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Variational Autoencoders (VAEs): A Complete Tutorial
 <p></p>
</div>

## [![CGANS](https://img.shields.io/badge/CGANs-Conditional_GAN-blue?style=for-the-badge&logo=github)](../posts/ConditionalGAN)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Python implementation of a Conditional Generative Adversarial Network (cGAN) using PyTorch.
 <p></p>
</div>


## [![GANS](https://img.shields.io/badge/GAN-Generative_Adversarial_Networks-blue?style=for-the-badge&logo=github)](../posts/GAN)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Generative Adversarial Networks (GANs) are composed of two neural networks:
* A **Generator (G)**: learns to generate fake samples $G(z)$ from random noise $z \sim p(z)$
* A **Discriminator (D)**: learns to classify samples as real (from data) or fake (from the generator)
* 
These networks are trained in a two-player minimax game.
  <p></p>
</div>

## [![SubPixelConv](https://img.shields.io/badge/SubPixelConv-Pixel_Shuffle-blue?style=for-the-badge&logo=github)](../posts/SubPixelConv)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
**Sub-pixel convolution** (also known as **pixel shuffle**) is a technique primarily used for **image super-resolution** and other upsampling tasks in deep learning. Instead of upsampling via interpolation or transposed convolution, it learns to generate a **high-resolution image** from a low-resolution feature map by **reorganizing the channels**.
  <p></p>
</div>

## [![FeatureHierarchy](https://img.shields.io/badge/FeatureHierarchy-Feature_Evolution_Along_DNN-blue?style=for-the-badge&logo=github)](../posts/FeatureHierarchy)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
This note explains how **features evolve across layers in deep neural networks** (especially CNNs), and how **fine-grained features** emerge and are preserved or enhanced for tasks like fine-grained classification, detection, and facial recognition.
  <p></p>
</div>


## [![Distillation](https://img.shields.io/badge/Distillation-grey?style=for-the-badge&logo=github)](../posts/Distillation)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Generalization issue with Distillation
 <p></p>
</div>


## [![MaskRCNN](https://img.shields.io/badge/MaskRCNN-Instancce_Segmentation-blue?style=for-the-badge&logo=github)](../posts/MaskRCNN)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
This tutorial is written to provide an extensive understanding of the Mask R-CNN architecture by dissecting every individual component involved in its pipeline.
</div>

## [![SSD](https://img.shields.io/badge/SSD-Single_Shot_Object_Detector-blue?style=for-the-badge&logo=github)](../posts/ssd)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Single shot object detector
</div>


