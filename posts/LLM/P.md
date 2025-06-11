[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../../)

## [![GenAI](https://img.shields.io/badge/GenAI-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../../main_page/GenAI)

# Kernel-Based Attention and the Performer: Full Tutorial

Performers are Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attentionkernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. Performer demonstrates competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

## 1. What Is a Kernel?

A **kernel** is a function that computes similarity between two inputs in a possibly high- or infinite-dimensional space without explicitly transforming the data. Formally:

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H} \tag{1}
$$

where:

* $\phi: \mathbb{R}^d \rightarrow \mathcal{H}$ is a feature map,
* $\mathcal{H}$ is a (possibly infinite) Hilbert space,
* $\langle \cdot, \cdot \rangle$ is the inner product in $\mathcal{H}$.

### What is a Shift-Invariant Kernel?

A **shift-invariant kernel** is one that depends only on the difference between its inputs, not on their absolute location:

$$
k(x, x') = k(x - x') = k(\delta) \tag{2}
$$

Such kernels are translation-invariant, meaning the similarity between two inputs does not change if both are shifted by the same amount. The **RBF kernel** is a prime example:

$$
k(x - x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right) \tag{3}
$$

---

## 2. Random Fourier Features (RFF)

Bochner's theorem tells us:

> A continuous, shift-invariant, positive-definite kernel is the Fourier transform of a non-negative probability distribution.

Therefore:

$$
k(x, x') = \int e^{i\omega^\top(x - x')} p(\omega) \, d\omega = \mathbb{E}_{\omega \sim p(\omega)}[e^{i\omega^\top x} e^{-i\omega^\top x'}] \tag{4}
$$

Using Euler's identity:

$$
e^{i\theta} = \cos \theta + i \sin \theta \tag{5}
$$
we get:

$$
k(x, x') = \mathbb{E}_\omega[\cos(\omega^\top x - \omega^\top x')] = \mathbb{E}_\omega[\cos(\omega^\top x) \cos(\omega^\top x') + \sin(\omega^\top x) \sin(\omega^\top x')] \tag{6}
$$

This yields a 2D feature map:

$$
\phi_\omega(x) = \begin{bmatrix} \cos(\omega^\top x) \\ \sin(\omega^\top x) \end{bmatrix}, \quad k(x, x') \approx \frac{1}{D} \sum_{j=1}^D \phi_{\omega_j}(x)^\top \phi_{\omega_j}(x') \tag{7}
$$

### Why We Can Omit the Sine Term

To reduce dimensionality and variance, we instead use a **random phase shift**:

$$
\cos(\omega^\top x + b) \cos(\omega^\top x' + b) = \cos(\omega^\top x - \omega^\top x') \tag{8}
$$
by sampling $b \sim \text{Uniform}[0, 2\pi]$. This yields:

$$
\phi(x) = \sqrt{\frac{2}{D}} \begin{bmatrix} \cos(\omega_1^\top x + b_1) \\ \vdots \\ \cos(\omega_D^\top x + b_D) \end{bmatrix} \tag{9}
$$

Then the kernel becomes:

$$
k(x, x') \approx \phi(x)^\top \phi(x') \tag{10}
$$
For the **RBF kernel**:

$$
p(\omega) = \mathcal{N}(0, \sigma^{-2} I), \quad \omega_j \sim \mathcal{N}(0, \sigma^{-2} I) \tag{11}
$$

### Infinite-Dimensional Nature of the RBF Kernel

$$
k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right) \tag{12}
$$

This kernel cannot be represented by a finite-dimensional $\phi(x)$, and thus must be approximated with RFF.

---

## 3. Performer: FAVOR+ Attention

The standard softmax attention formulation is:

$$
\text{Attention}(Q, K, V) = D^{-1} A V, \quad A = \exp\left(\frac{QK^\top}{\sqrt{d}}\right), \quad D = \text{diag}(A \mathbf{1}_L) \tag{13}
$$


This is computationally expensive due to the $O(n^2)$ complexity from computing all pairwise dot-products in $QK^\top$.

Performer replaces this with a kernel-based approximation using **Random Fourier Features**:

$$
\text{Attention}(Q, K, V) \approx \hat{D}^{-1} \left(\phi(Q)(\phi(K)^\top V)\right), \quad \hat{D} = \text{diag}(\phi(Q)(\phi(K)^\top \mathbf{1}_L)) \tag{14}
$$

Let the feature map be:

$$
\phi(x) = \sqrt{\frac{2}{D}} [\cos(\omega_1^\top x + b_1), ..., \cos(\omega_D^\top x + b_D)]^\top \tag{15}
$$

### Step-by-Step Matrix Operations

Let:

* $\tilde{Q} = \phi(Q) \in \mathbb{R}^{n \times D}$
* $\tilde{K} = \phi(K) \in \mathbb{R}^{n \times D}$
* $Z = \tilde{K}^\top V \in \mathbb{R}^{D \times d}$

Then:

* Multiply $\tilde{Q} Z$ to get numerator $A \in \mathbb{R}^{n \times d}$
* Normalize by:

$$
\text{norm}_i = \tilde{Q}_i^\top (\tilde{K}^\top \mathbf{1}) \tag{16}
$$


Final output:
$$
\text{PerformerAttn}(Q, K, V)_i = \frac{\tilde{Q}_i^\top (\tilde{K}^\top V)}{\tilde{Q}_i^\top (\tilde{K}^\top \mathbf{1})} \tag{17}
$$

### Minimal PyTorch-Style Implementation

```python
import torch
import math

def random_features(x, omega, b):
    # x: (batch, seq_len, dim), omega: (dim, n_features), b: (n_features,)
    projection = torch.matmul(x, omega) + b  # (batch, seq_len, n_features)
    return math.sqrt(2.0 / omega.shape[1]) * torch.cos(projection)

def performer_attention(Q, K, V, omega, b, eps=1e-6):
    Q_prime = random_features(Q, omega, b)  # (B, L, D)
    K_prime = random_features(K, omega, b)  # (B, L, D)

    KV = torch.einsum('bld,blh->bdh', K_prime, V)     # (B, D, H)
    Z = torch.einsum('bld,bdh->blh', Q_prime, KV)     # (B, L, H)
    normalizer = torch.einsum('bld,bd->bl', Q_prime, K_prime.sum(dim=1)) + eps  # (B, L)
    return Z / normalizer.unsqueeze(-1)  # (B, L, H)
```

This approximation reduces time and space complexity from $O(n^2)$ to $O(n)$, while retaining expressive power.

---
