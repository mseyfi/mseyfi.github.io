[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../main_page/GenAI)


# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**Introduction**

"Learning Transferable Visual Models From Natural Language Supervision" is a groundbreaking paper by OpenAI that introduces **CLIP** (Contrastive Language-Image Pre-training). CLIP learns visual concepts from natural language supervision by jointly training an image encoder and a text encoder to predict the correct pairings of images and texts.

In this tutorial, we'll explore:

- How inputs for the vision and text parts are preprocessed using code.
- What tokenization means.
- What a corpus is.
- The input and output of the tokenizer and their shapes.
- The architectures of the vision and language models using code.
- The loss function used during training.
- How inference is performed using code.
- Sample applications with code examples.

---

## Table of Contents

1. [Input Preprocessing](#input-preprocessing)
   - [Vision Input Preprocessing](#vision-input-preprocessing)
   - [Text Input Preprocessing](#text-input-preprocessing)
2. [Model Architectures](#model-architectures)
   - [Vision Model Architecture](#vision-model-architecture)
   - [Language Model Architecture](#language-model-architecture)
3. [Loss Function](#loss-function)
   - [Contrastive Loss Explanation](#contrastive-loss-explanation)
   - [Loss Function Implementation](#loss-function-implementation)
4. [Inference](#inference)
   - [Performing Inference with Code](#performing-inference-with-code)
5. [Sample Applications](#sample-applications)
   - [Zero-Shot Image Classification](#zero-shot-image-classification)
   - [Image Retrieval](#image-retrieval)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Input Preprocessing

### Vision Input Preprocessing

The vision encoder requires images to be preprocessed into a consistent format.

**Steps:**

1. **Resize**: Adjust the image size to a fixed resolution.
2. **Center Crop**: Crop the central part of the image.
3. **Convert to Tensor**: Transform the image into a PyTorch tensor.
4. **Normalize**: Normalize the tensor using mean and standard deviation.

**Code Example:**

```python
import torch
from PIL import Image
from torchvision import transforms

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(224),          # Resize the image to 224 pixels on the shorter side
    transforms.CenterCrop(224),      # Crop the center 224x224 pixels
    transforms.ToTensor(),           # Convert PIL image to PyTorch tensor
    transforms.Normalize(            # Normalize the tensor
        mean=[0.48145466, 0.4578275, 0.40821073],   # Mean values for each channel (RGB)
        std=[0.26862954, 0.26130258, 0.27577711]    # Standard deviation for each channel (RGB)
    )
])

# Load and preprocess the image
image = Image.open("path_to_your_image.jpg")
image_input = preprocess(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
```

**Explanation:**

- **`transforms.Resize(224)`**: Resizes the shorter side of the image to 224 pixels, maintaining aspect ratio.
- **`transforms.CenterCrop(224)`**: Crops a 224x224 pixel square from the center of the image.
- **`transforms.ToTensor()`**: Converts the image to a tensor and scales pixel values to [0, 1].
- **`transforms.Normalize(mean, std)`**: Normalizes each channel of the image tensor.
- **`unsqueeze(0)`**: Adds a batch dimension; the shape becomes **[batch_size, channels, height, width]**.

**Tensor Shape After Preprocessing:**

- **`image_input`**: Shape **[1, 3, 224, 224]** (batch_size=1, channels=3, height=224, width=224)

### Text Input Preprocessing

The text encoder requires text to be tokenized.

**What is Tokenization?**

Tokenization is the process of breaking down text into smaller units called tokens, which can be words, subwords, or characters. In NLP models, tokens are converted into numerical IDs that the model can process.

**What is a Corpus?**

A corpus is a large collection of text used for training language models. In this context, the corpus refers to the set of text descriptions paired with images.

**Tokenization Process:**

1. **Tokenization**: Split the text into tokens using a tokenizer.
2. **Convert to Token IDs**: Map tokens to numerical IDs.
3. **Padding**: Pad sequences to a fixed length.
4. **Create Attention Masks** (if necessary): Indicate which tokens are actual data and which are padding.

**Code Example:**

```python
import clip

# Text descriptions
text_descriptions = ["a photo of a cat", "a picture of a dog"]

# Tokenize the text descriptions
text_tokens = clip.tokenize(text_descriptions)  # Shape: [batch_size, sequence_length]

print("Tokenized Text Shape:", text_tokens.shape)
```

**Explanation:**

- **`clip.tokenize()`**: Tokenizes and encodes text descriptions into token IDs, padding them to a fixed sequence length (e.g., 77 tokens).
- **`text_tokens`**: A tensor containing token IDs; shape is **[batch_size, sequence_length]**.

**Tensor Shape After Tokenization:**

- **`text_tokens`**: Shape **[2, 77]** (batch_size=2, sequence_length=77)

**Understanding the Output:**

- Each row in `text_tokens` corresponds to a text description.
- Each column represents a token position up to the maximum sequence length.
- The values are token IDs corresponding to tokens in the tokenizer's vocabulary.

---

## Model Architectures

### Vision Model Architecture

The vision encoder in CLIP can be a **ResNet** or a **Vision Transformer (ViT)**. We'll focus on the Vision Transformer.

**Vision Transformer (ViT) Architecture:**

- **Patch Embedding**: The image is divided into patches, each flattened and projected to an embedding space.
- **Position Embedding**: Adds positional information to each patch embedding.
- **Transformer Encoder**: Applies self-attention and feed-forward layers.
- **Projection**: Projects the output to a shared embedding space.

**Code Simplification:**

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size

        # Calculate the number of patches
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels=3,                # RGB channels
            out_channels=embed_dim,       # Embedding dimension
            kernel_size=patch_size,       # Size of each patch
            stride=patch_size             # Non-overlapping patches
        )
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))           # Shape: [1, 1, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # Shape: [1, num_patches + 1, embed_dim]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final projection layer
        self.proj = nn.Linear(embed_dim, 512)  # Project to 512-dimensional embedding

    def forward(self, x):
        # x shape: [batch_size, 3, image_size, image_size]
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # Shape: [batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt]
        x = x.flatten(2)         # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)    # Shape: [batch_size, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)                   # Shape: [batch_size, num_patches + 1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed  # Shape: [batch_size, num_patches + 1, embed_dim]

        # Transformer encoding
        x = x.permute(1, 0, 2)  # Shape: [num_patches + 1, batch_size, embed_dim]
        x = self.transformer(x) # Shape: [num_patches + 1, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # Shape: [batch_size, num_patches + 1, embed_dim]

        # Take the class token
        x = x[:, 0, :]          # Shape: [batch_size, embed_dim]

        # Final projection
        x = self.proj(x)        # Shape: [batch_size, 512]

        return x
```

**Explanation:**

- **Patch Embedding**: The Conv2d layer divides the image into patches and maps them to embeddings.
- **Class Token**: A learnable parameter that represents the whole image.
- **Positional Embedding**: Encodes positional information of patches.
- **Transformer Encoder**: Processes the sequence of patch embeddings.
- **Final Projection**: Maps the embeddings to a 512-dimensional space.

**Tensor Shapes During Forward Pass:**

- Input `x`: Shape **[batch_size, 3, 224, 224]**
- After patch embedding: **[batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt]**
- After flattening and transpose: **[batch_size, num_patches, embed_dim]**
- After adding class token: **[batch_size, num_patches + 1, embed_dim]**
- After positional embedding: **[batch_size, num_patches + 1, embed_dim]**
- After transformer: **[batch_size, num_patches + 1, embed_dim]**
- After taking class token: **[batch_size, embed_dim]**
- After projection: **[batch_size, 512]**

### Language Model Architecture

The text encoder is a Transformer-based model, similar to GPT.

**Language Transformer Architecture:**

- **Token Embedding**: Maps tokens to embeddings.
- **Position Embedding**: Adds positional information.
- **Transformer Encoder**: Processes the token sequence.
- **Projection**: Projects the output to a shared embedding space.

**Code Simplification:**

```python
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_length=77, embed_dim=512, depth=12, num_heads=8):
        super(TextTransformer, self).__init__()

        # Token and positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)           # Shape: [vocab_size, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))  # Shape: [1, max_seq_length, embed_dim]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final normalization and projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 512)  # Project to 512-dimensional embedding

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        batch_size, seq_length = x.size()

        # Token embedding
        x = self.token_embed(x)    # Shape: [batch_size, seq_length, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, :seq_length, :]  # Shape: [batch_size, seq_length, embed_dim]

        # Transformer encoding
        x = x.permute(1, 0, 2)     # Shape: [seq_length, batch_size, embed_dim]
        x = self.transformer(x)    # Shape: [seq_length, batch_size, embed_dim]
        x = x.permute(1, 0, 2)     # Shape: [batch_size, seq_length, embed_dim]

        # Take the representation of the last token
        x = x[torch.arange(batch_size), x.argmax(dim=1)]  # Shape: [batch_size, embed_dim]

        # Final layer normalization and projection
        x = self.ln_final(x)       # Shape: [batch_size, embed_dim]
        x = self.proj(x)           # Shape: [batch_size, 512]

        return x
```

**Explanation:**

- **Token Embedding**: Converts token IDs to embeddings.
- **Positional Embedding**: Adds positional information to each token embedding.
- **Transformer Encoder**: Processes the sequence of token embeddings.
- **Final Layer Norm and Projection**: Normalizes and projects the output to a 512-dimensional space.

**Tensor Shapes During Forward Pass:**

- Input `x`: Shape **[batch_size, seq_length]**
- After token embedding: **[batch_size, seq_length, embed_dim]**
- After positional embedding: **[batch_size, seq_length, embed_dim]**
- After transformer: **[batch_size, seq_length, embed_dim]**
- After selecting token representation: **[batch_size, embed_dim]**
- After layer norm and projection: **[batch_size, 512]**

---

## Loss Function

### Contrastive Loss Explanation

CLIP uses a contrastive loss to train the model. The goal is to bring matching image-text pairs closer in the embedding space while pushing non-matching pairs apart.

**Process:**

1. **Normalize Embeddings**: Ensure embeddings have unit length.
2. **Compute Similarity Matrix**: Calculate cosine similarities between all image and text embeddings.
3. **Apply Temperature Scaling**: Adjust the distribution sharpness.
4. **Compute Cross-Entropy Loss**: For both image-to-text and text-to-image directions.

### Loss Function Implementation

**Code Example:**

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Compute the CLIP contrastive loss.

    Parameters:
    - image_embeddings: [batch_size, embedding_dim]
    - text_embeddings: [batch_size, embedding_dim]
    - temperature: scalar

    Returns:
    - loss: scalar
    """

    # Normalize embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # Shape: [batch_size, embedding_dim]
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)     # Shape: [batch_size, embedding_dim]

    # Compute cosine similarity matrix
    logits_per_image = image_embeddings @ text_embeddings.t() / temperature  # Shape: [batch_size, batch_size]
    logits_per_text = logits_per_image.t()                                   # Shape: [batch_size, batch_size]

    # Labels are the indices of the diagonal
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size).to(image_embeddings.device)            # Shape: [batch_size]

    # Cross-entropy loss for image-to-text
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    # Cross-entropy loss for text-to-image
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    # Total loss
    loss = (loss_i2t + loss_t2i) / 2

    return loss
```

**Explanation:**

- **Normalization**: Ensures that embeddings lie on the unit hypersphere.
- **Similarity Matrix**: Computes cosine similarities between all image-text pairs.
- **Temperature Scaling**: Controls the sharpness of the distribution.
- **Labels**: Correct pairs are on the diagonal; labels are indices from 0 to batch_size - 1.
- **Cross-Entropy Loss**: Calculated for both directions and averaged.

---

## Inference

### Performing Inference with Code

**Code Example:**

```python
import torch
import clip
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess the image
image = preprocess(Image.open("path_to_your_image.jpg")).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# Prepare the text descriptions
text_descriptions = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_tokens = clip.tokenize(text_descriptions).to(device)  # Shape: [3, 77]

# Encode the image and text
with torch.no_grad():
    image_features = model.encode_image(image)  # Shape: [1, 512]
    text_features = model.encode_text(text_tokens)  # Shape: [3, 512]

    # Normalize the features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)  # Shape: [1, 512]
    text_features = text_features / text_features.norm(dim=1, keepdim=True)     # Shape: [3, 512]

    # Compute similarity scores
    similarity = image_features @ text_features.t()  # Shape: [1, 3]

    # Apply softmax to get probabilities
    probs = similarity.softmax(dim=-1)  # Shape: [1, 3]

# Output the probabilities
print("Probabilities:", probs.cpu().numpy())
```

**Explanation:**

- **Image Preprocessing**: The image is preprocessed and reshaped to **[1, 3, 224, 224]**.
- **Text Tokenization**: Text descriptions are tokenized to **[3, 77]**.
- **Encoding**: Image and text are encoded to **[1, 512]** and **[3, 512]** respectively.
- **Normalization**: Embeddings are normalized.
- **Similarity Calculation**: The dot product computes similarities between the image and each text description.
- **Softmax**: Converts similarities to probabilities.

---

## Sample Applications

### Zero-Shot Image Classification

**Task**: Classify images without explicit training on those classes.

**Code Example:**

```python
import torch
import clip
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List of class names
class_names = ["cat", "dog", "bird", "car", "plane"]

# Create text descriptions
text_descriptions = [f"a photo of a {classname}" for classname in class_names]
text_tokens = clip.tokenize(text_descriptions).to(device)  # Shape: [num_classes, 77]

# Preprocess the image
image = preprocess(Image.open("path_to_your_image.jpg")).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# Encode the image and text
with torch.no_grad():
    image_features = model.encode_image(image)  # Shape: [1, 512]
    text_features = model.encode_text(text_tokens)  # Shape: [num_classes, 512]

    # Normalize the features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)  # Shape: [1, 512]
    text_features = text_features / text_features.norm(dim=1, keepdim=True)     # Shape: [num_classes, 512]

    # Compute similarity scores
    similarity = image_features @ text_features.t()  # Shape: [1, num_classes]

    # Apply softmax to get probabilities
    probs = similarity.softmax(dim=-1)  # Shape: [1, num_classes]

# Get the predicted class
predicted_class = class_names[probs.argmax().item()]
print(f"Predicted class: {predicted_class}")
```

**Explanation:**

- **Class Names**: Define the classes you want to classify.
- **Text Descriptions**: Create phrases that describe each class.
- **Encoding and Similarity**: Same as before.
- **Prediction**: The class with the highest probability is selected.

### Image Retrieval

**Task**: Retrieve images that match a given text query.

**Code Example:**

```python
import torch
import clip
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List of image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
images = [preprocess(Image.open(p)).unsqueeze(0) for p in image_paths]  # Each image shape: [1, 3, 224, 224]
image_inputs = torch.cat(images, dim=0).to(device)  # Shape: [num_images, 3, 224, 224]

# Text query
text_query = "a beautiful landscape"
text_tokens = clip.tokenize([text_query]).to(device)  # Shape: [1, 77]

# Encode images and text
with torch.no_grad():
    image_features = model.encode_image(image_inputs)  # Shape: [num_images, 512]
    text_features = model.encode_text(text_tokens)     # Shape: [1, 512]

    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)  # Shape: [num_images, 512]
    text_features = text_features / text_features.norm(dim=1, keepdim=True)     # Shape: [1, 512]

    # Compute similarities
    similarities = image_features @ text_features.t()  # Shape: [num_images, 1]

# Get the most similar image
most_similar_idx = similarities.argmax().item()
most_similar_image_path = image_paths[most_similar_idx]
print(f"Most similar image: {most_similar_image_path}")
```

**Explanation:**

- **Image Loading**: All images are preprocessed and stacked.
- **Text Query**: The user's query is tokenized.
- **Encoding**: Both images and text are encoded and normalized.
- **Similarity Calculation**: Computes similarity between each image and the text query.
- **Retrieval**: The image with the highest similarity score is selected.

---

## Addendum
Let's break down the **tensor shapes in CLIP's Vision Transformer (ViT)** encoder step-by-step — from raw image to final image embedding. I'll use **ViT-B/32** as the base example and show how tensors flow.

---

## 🖼️ Input Image

Assume:

* Input image: `img`
* Shape:

  $$
  \text{img} \in \mathbb{R}^{B \times 3 \times H \times W}
  $$

  * $B$: batch size
  * $H = W = 224$
  * $3$: RGB channels

---

## 🧩 1. Patch Embedding

* Patch size $P = 32$
* Number of patches per image:

  $$
  N = \frac{H}{P} \cdot \frac{W}{P} = \frac{224}{32} \cdot \frac{224}{32} = 7 \cdot 7 = 49
  $$

**Operation**:

* Apply a Conv2d with:

  * `in_channels = 3`
  * `out_channels = D = 512`
  * `kernel_size = 32`, `stride = 32`

**Output shape after Conv2d**:

$$
\text{patches} \in \mathbb{R}^{B \times 512 \times 7 \times 7}
$$

**Flatten patches**:

$$
\text{patches} \rightarrow \mathbb{R}^{B \times 49 \times 512}
$$

---

## 🔤 2. Add Class Token

* A learnable token of shape $\mathbb{R}^{1 \times 1 \times 512}$
* Concatenated to the beginning:

$$
\text{tokens} \in \mathbb{R}^{B \times 50 \times 512}
$$

* 49 patches + 1 class token = 50 tokens

---

## 📍 3. Add Positional Embedding

* Positional embedding:

  $$
  \text{pos\_embed} \in \mathbb{R}^{1 \times 50 \times 512}
  $$

* Added element-wise:

$$
\text{tokens} \leftarrow \text{tokens} + \text{pos\_embed}
$$

Shape remains:

$$
\text{tokens} \in \mathbb{R}^{B \times 50 \times 512}
$$

---

## 🧠 4. Transformer Encoder

* 12 layers (ViT-B/32) of Multi-head Self-Attention (MSA) + MLP

* Input:

  $$
  \text{tokens} \in \mathbb{R}^{B \times 50 \times 512}
  $$

* Output (same shape):

  $$
  \text{encoded} \in \mathbb{R}^{B \times 50 \times 512}
  $$

---

## 🎯 5. Extract Class Token & Projection

* Take only the **first token** (class token):

  $$
  \text{class\_token} \in \mathbb{R}^{B \times 512}
  $$

* Apply a linear projection head (optional in CLIP):

  $$
  \text{image\_embedding} \in \mathbb{R}^{B \times 512}
  $$

* Normalize:

  $$
  \text{image\_embedding} \leftarrow \frac{\text{image\_embedding}}{\|\cdot\|}
  $$

---

## 🔢 Summary Table

| Stage                 | Tensor Shape                       | Description                                  |
| --------------------- | ---------------------------------- | -------------------------------------------- |
| Input Image           | $B \times 3 \times 224 \times 224$ | RGB image                                    |
| After Patch Embedding | $B \times 49 \times 512$           | 49 flattened patch tokens                    |
| Add Class Token       | $B \times 50 \times 512$           | 49 patches + 1 class token                   |
| Add Positional Embed  | $B \times 50 \times 512$           | same shape after addition                    |
| Transformer Output    | $B \times 50 \times 512$           | encoded tokens                               |
| Extract Class Token   | $B \times 512$                     | final image representation                   |
| Normalize             | $B \times 512$                     | L2-normalized embedding for contrastive loss |

---


## Conclusion

In this tutorial, we've explored:

- **Input Preprocessing**: How images and texts are preprocessed before being fed into the model.
- **Tokenization**: The process of converting text into tokens and numerical IDs.
- **Model Architectures**: The design of the vision and language models in CLIP.
- **Loss Function**: How the contrastive loss is implemented to train the model.
- **Inference**: Performing inference to get predictions from the model.
- **Sample Applications**: Practical examples like zero-shot image classification and image retrieval.

Understanding these components provides a solid foundation for working with CLIP and leveraging its capabilities in various applications.

---

## References

- **Original Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **CLIP GitHub Repository**: [OpenAI CLIP](https://github.com/openai/CLIP)
- **PyTorch Documentation**: [PyTorch](https://pytorch.org/docs/stable/index.html)
- **Vision Transformer Paper**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---

**Note**: The code examples provided are for educational purposes and may require additional context or adjustments to run in a specific environment. Always refer to official documentation for the most accurate information.
