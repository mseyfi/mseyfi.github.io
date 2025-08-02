Here’s a revised table summarizing **state-of-the-art industrial anomaly detection methods**, focusing on **detection accuracy**, **inference speed**, and key **pros and cons**. Where exact numbers weren’t public, I've described relative performance based on benchmark studies.

---

### 🏭 SOTA in Industrial Anomaly Detection

| Method        | Year | Dataset       | Accuracy / AUROC                           | Inference Speed                                                  | Pros                                                                                                           | Cons                                                        |
| ------------- | ---- | ------------- | ------------------------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **PatchCore** | 2022 | MVTec‑AD      | SOTA image & pixel                         | Real-time with coreset; faster than PaDiM ([CVF Open Access][1]) | Extremely accurate; flexible (no spatial alignment); customizable speed–accuracy tradeoff via coreset sampling | Memory-heavy without subsampling; slower if no coreset used |
| **PaDiM**     | 2020 | MVTec‑AD, STC | SOTA localization                          | Moderate (CNN backbone) ([arXiv][2], [tomomi-research.com][3])   | Accurate localization; probabilistic modeling via Gaussian                                                     | Requires spatial alignment; slower than PatchCore           |
| **FAPM**      | 2022 | Industrial    | Comparable to SOTA                         | Designed for real-time ([arXiv][4])                              | Efficient via patch-wise memory; fast adaptive sampling                                                        | Newer; fewer evaluation benchmarks available                |
| **EasyNet**   | 2023 | 3D (MVTec‑3D) | \~92.6% AUROC                              | \~94.6 FPS on V100 GPU ([arXiv][5])                              | Memory-bank-free; lightweight and fast; RGB-D fusion                                                           | Only 3D; performance figures only for one setting           |
| **HyADS**     | 2025 | Industrial    | Increased F1 vs others                     | Faster than FastAD and EdgeCore ([mdpi.com][6])                  | Hybrid architecture combining texture, reconstruction, segmentation                                            | Yet to establish leaderboard dominance; more details needed |
| **M3DM**      | 2023 | MVTec‑3D AD   | SOTA detection & segmentation (multimodal) | (Not specified) ([arXiv][7])                                     | Handles RGB + 3D; robust fusion and alignment                                                                  | No speed metrics available; complexity of fusion            |

---

### 🔎 Insights

* **PatchCore** remains a gold standard in performance for 2D industrial anomaly detection. Its ability to subsample and use approximate search makes it highly adaptable to speed or accuracy demands.
* **PaDiM** shines in precise localization but performs best on aligned datasets and lacks the flexibility of PatchCore.
* **FAPM** shows promise in achieving a sweet spot between speed and accuracy via clever memory sampling.
* In the **3D domain**, **EasyNet** and **M3DM** are newer but powerful contenders—EasyNet for speed and simplicity in RGB-D settings, M3DM for multimodal robustness.
* **HyADS** is a recent hybrid approach showing strong F1-score gains and faster inference, although broader benchmarking remains forthcoming.

---

### ✅ Recommendation

* For **2D industrial inspection**, **PatchCore** is a safe bet for top accuracy, especially if you can afford its resource requirements—or use coreset techniques to accelerate inference.
* Choose **PaDiM** if spatial structure and alignment are reliable and you need precise localization.
* **FAPM** may be ideal when **real-time throughput** is critical.
* For **3D data**, consider **EasyNet** if you value speed or **M3DM** if you need multimodal integration.
* Keep an eye on **HyADS** for future production-ready hybrid solutions.

---

If you'd like actual quantitative metrics (e.g., FPS, ms/image) for any of these or details on other methods like CFlow-AD, Fastrecon, or diffusion-based models, I can dig deeper!

[1]: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf?utm_source=chatgpt.com "Towards Total Recall in Industrial Anomaly Detection"
[2]: https://arxiv.org/abs/2011.08785?utm_source=chatgpt.com "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"
[3]: https://www.tomomi-research.com/en/archives/2968?utm_source=chatgpt.com "[Tech Blog] Overview of Anomaly Detection Technologies"
[4]: https://arxiv.org/abs/2211.07381?utm_source=chatgpt.com "FAPM: Fast Adaptive Patch Memory for Real-time Industrial Anomaly Detection"
[5]: https://arxiv.org/abs/2307.13925?utm_source=chatgpt.com "EasyNet: An Easy Network for 3D Industrial Anomaly Detection"
[6]: https://www.mdpi.com/2079-9292/14/11/2250?utm_source=chatgpt.com "HyADS: A Hybrid Lightweight Anomaly Detection ..."
[7]: https://arxiv.org/abs/2303.00601?utm_source=chatgpt.com "Multimodal Industrial Anomaly Detection via Hybrid Fusion"


# PaDim

Absolutely. Let's break down **PaDiM (Patch Distribution Modeling)** — one of the best-performing and most efficient methods for **unsupervised anomaly detection and localization**, particularly suited for **industrial vision tasks** like PCB defect detection or surface scratch detection.

---

## 🧠 Intuition Behind PaDiM

**Key idea**:
Instead of modeling the whole image as a single distribution, **model the distribution of CNN features at each spatial patch location individually** using a **multivariate Gaussian**.

### Why this works:

* Normal patches at the **same spatial location** (e.g., resistor at row 7, col 12) tend to have **consistent feature distributions**
* Anomalous patches (e.g., missing or faulty parts) deviate from that local distribution
* Using Mahalanobis distance allows **scale-invariant outlier detection**

---

## 🏗️ Architecture and Flow

### 1. **Feature Extraction (Pretrained CNN)**

* Use a **pretrained CNN** (e.g., WideResNet50 or ResNet18)
* Select **intermediate layers** (e.g., `layer1`, `layer2`, `layer3`) and concatenate spatial features
* This gives you a **feature map** $f(x) \in \mathbb{R}^{C \times H \times W}$

### 2. **Fit Gaussian at Each Patch**

For each spatial location $(i, j)$ in the feature map:

* Collect feature vectors $f_{(i,j)} \in \mathbb{R}^C$ from all **normal training images**
* Estimate:

  * Mean: $\mu_{(i,j)} \in \mathbb{R}^C$
  * Covariance: $\Sigma_{(i,j)} \in \mathbb{R}^{C \times C}$

This gives a **Gaussian distribution per location**:

$$
f_{(i,j)} \sim \mathcal{N}(\mu_{(i,j)}, \Sigma_{(i,j)})
$$

To reduce memory and overfitting, apply **random projection** to reduce feature dim $C \to C'$ (e.g., 512 → 100)

### 3. **Anomaly Score at Inference**

At test time:

* Extract feature map of test image
* For each patch $(i, j)$, compute the **Mahalanobis distance** to its corresponding training distribution:

$$
d_{(i,j)} = \sqrt{(f_{(i,j)} - \mu_{(i,j)})^T \Sigma_{(i,j)}^{-1} (f_{(i,j)} - \mu_{(i,j)})}
$$

### 4. **Anomaly Heatmap and Image Score**

* Build a 2D **heatmap** from all $d_{(i,j)}$ values
* Optionally upsample to input resolution
* Image-level score = **max distance** or **average distance**

---

## 🔢 Summary of Key Components

| Component        | Description                                     |
| ---------------- | ----------------------------------------------- |
| Backbone         | Pretrained CNN (e.g., ResNet50)                 |
| Feature Layers   | Intermediate layers like `layer1` + `layer2`    |
| Patch modeling   | Gaussian $\mathcal{N}(\mu, \Sigma)$ per (i,j)   |
| Distance metric  | Mahalanobis distance                            |
| Training data    | Only **normal images**                          |
| Inference output | Per-pixel anomaly heatmap + image anomaly score |

---

## 🧮 Mathematical Formulation

Let:

* $f_{(i,j)} \in \mathbb{R}^{C}$: feature vector at location $(i,j)$
* $\mu_{(i,j)}, \Sigma_{(i,j)}$: mean and covariance estimated from training data

Then, the anomaly score at that patch is:

$$
\text{score}_{(i,j)} = D_M(f_{(i,j)}) = \sqrt{(f_{(i,j)} - \mu_{(i,j)})^\top \Sigma_{(i,j)}^{-1}(f_{(i,j)} - \mu_{(i,j)})}
$$

---

## 🧪 Advantages of PaDiM

| Feature                     | Benefit                                                       |
| --------------------------- | ------------------------------------------------------------- |
| Patch-wise modeling         | Precise localization of small anomalies (e.g., missing parts) |
| No training                 | Only involves feature extraction + statistics computation     |
| Unsupervised                | Needs only **normal** (non-defective) training samples        |
| Fast inference              | Just CNN forward + Mahalanobis distance                       |
| Compact memory (with proj.) | Works well even with 100-dim features                         |

---

## 🐍 PyTorch-like Pseudocode (Training)

```python
# Training phase: fit Gaussians
features = []  # list of (C, H, W) features from normal images
for img in normal_images:
    feat = model.forward_features(img)  # pretrained CNN
    features.append(feat)

features = torch.stack(features)  # (N, C, H, W)

mu = features.mean(dim=0)  # (C, H, W)
sigma = compute_covariance_matrix(features)  # (C, C, H, W)
```

---

## 🔬 Inference Phase

```python
feat_test = model.forward_features(test_img)  # (C, H, W)
score_map = torch.zeros(H, W)
for i in range(H):
    for j in range(W):
        f_ij = feat_test[:, i, j]
        score_map[i, j] = mahalanobis(f_ij, mu[:, i, j], sigma[:, :, i, j])
```

---

## 📊 Results on MVTec AD (benchmark)

| Category          | AUROC (%) |
| ----------------- | --------- |
| **Mean over all** | \~98.2    |
| PCB               | \~99.2    |
| Screw             | \~99.0    |
| Cable             | \~98.7    |

Very strong performer, especially for structured, repetitive components like PCBs.

---

## ✅ Best Use Cases

* Electronics inspection (PCB, IC, connectors)
* Industrial defect detection (scratches, cracks, misalignments)
* Assembly verification

---

# patchCore

Absolutely. **PatchCore** is a **state-of-the-art anomaly detection** method for industrial images that improves both **accuracy and inference speed** using a technique called **coreset sampling**. Let’s go through how PatchCore works, with an in-depth focus on **coreset construction**.

---

## 🧠 High-Level Overview of PatchCore

PatchCore combines:

1. **Backbone feature extractor** (e.g., ResNet or ViT)
2. **Patch-wise feature collection**
3. **Coreset sampling** from normal patches
4. **Test-time nearest neighbor search**
5. **Anomaly localization** (via patch-wise distances)

---

## 🔍 PatchCore Inference Pipeline (Step-by-step)

### **Step 1: Feature Extraction**

* For each training image (normal only), extract **intermediate features** (e.g., from ResNet’s `layer2` or ViT tokens).
* Each image becomes a **set of patch-level features**:

  $$
  F = \{f_1, f_2, ..., f_N\},\quad f_i \in \mathbb{R}^d
  $$
* Collect features from **all training images** into a large memory bank:

  $$
  \mathcal{F}_{\text{train}} \in \mathbb{R}^{M \times d},\quad M = \text{total patches}
  $$

---

### **Step 2: Coreset Sampling**

This step **reduces the size** of the memory bank while **preserving diversity and coverage**, to accelerate nearest neighbor search at test time.

#### 🧠 Why?

* Without coreset: memory = 10,000 images × 196 patches = \~2 million vectors!
* k-NN search is expensive in large memory — even with Faiss.
* Goal: Select **a small representative subset** (e.g., 1%–10%) of patches that still **cover** the feature space well.

---

## 🔧 Coreset Sampling in PatchCore

### ✴️ Algorithm: **Greedy k-Center Coreset**

> Iteratively select diverse points such that **every point in the full set is close to at least one point in the coreset**.

Let’s denote:

* Full patch feature set: `F_full ∈ ℝ^{M × d}`
* Coreset subset: `F_core ⊂ F_full`, initialized empty

### 🚶 Steps:

1. **Randomly pick the first point**:
   Add one random point from `F_full` to `F_core`.

2. **Iteratively select the point farthest away from the current coreset**:
   For all points `x ∈ F_full`, compute:

   $$
   d(x, F_{\text{core}}) = \min_{y \in F_{\text{core}}} \|x - y\|_2
   $$

   Select the point `x*` with the **maximum distance** from the current coreset:

   $$
   x^* = \arg\max_{x \in F_{\text{full}}} d(x, F_{\text{core}})
   $$

3. **Add x\*** to the coreset:

   $$
   F_{\text{core}} \gets F_{\text{core}} \cup \{x^*\}
   $$

4. **Repeat until the target size (e.g., 1% of M) is reached.**

---

### 📏 Complexity

* Time: $\mathcal{O}(k \cdot M \cdot d)$
  where `k` is the number of coreset points

* This is still costly **offline**, but **accelerates inference** significantly.

* In practice, implemented efficiently using **batch vectorized distance computations** (e.g., with Faiss or PyTorch).

---

## 🚀 Inference Phase

At test time:

* For each patch in the test image:

  * Compute its distance to the **nearest coreset patch** (e.g., Euclidean or cosine distance)
  * Use the **minimum distance** as the **anomaly score**

* For localization:

  * Map patch-wise scores to a grid (e.g., 14×14) and upsample to original resolution (e.g., 224×224)

---

## 📊 Summary Table

| Step               | Operation                                       | Output                            |
| ------------------ | ----------------------------------------------- | --------------------------------- |
| Feature extraction | Backbone on normal images                       | Patch features `F_full ∈ ℝ^{M×d}` |
| Coreset sampling   | k-center greedy algorithm                       | Subset `F_core ⊂ F_full`          |
| Anomaly scoring    | Nearest neighbor from test patches to `F_core`  | Patch-level distances             |
| Localization       | Interpolate patch grid back to image resolution | Heatmap                           |

---

## ✅ Advantages of Coreset Sampling

| Benefit                   | Explanation                                              |
| ------------------------- | -------------------------------------------------------- |
| **Speed**                 | Smaller memory bank → faster k-NN search                 |
| **Memory efficiency**     | Less RAM / GPU memory needed                             |
| **Maintains performance** | Accuracy comparable to full memory when k-center is used |
| **No retraining**         | Fully post-processing; feature extractor remains fixed   |

---

## 📌 Example

Say you extract `2 million` patch features from the training set.

* Coreset ratio: `1%` → only `20,000` kept
* You now run k-NN search for each test patch only on `20,000` vectors instead of `2M`
* Result: **up to 20× faster inference**, same accuracy (as shown in PatchCore paper)

---

## 🧪 Reference

* **Roth et al., CVPR 2022**
  *"Towards Total Recall in Industrial Anomaly Detection"*
  [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)

---

