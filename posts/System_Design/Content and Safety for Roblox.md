## Content and Safety for Roblox Image



1. How much time a kid spends with a stranger
2. Age monitoring
3. Sexual Misconduct
4. Control Image sharing
5. Monitor text for critical hazards language
6. Moderate text on the platform
7. Screen time
8. Send notification to parents if suspicious messages and harmful content is detected.
9. Voice and text in the chatrooms should be monitored.
10. Hazardous links.


**Roblox Content & Safety ML System Design: Image Moderation**

---

### 1. Interview Kickoff: Clarifying Questions

Before starting system design, ask the interviewer:

#### Scope & Goals

* What types of policy violations should the system detect? (e.g. nudity, violence, hate symbols)
* Is the moderation real-time or batch?
* Is the system fully automated or is there a human-in-the-loop?

#### Constraints & Metrics

* Expected image upload rate (throughput)?
* Target latency for moderation decision?
* Which is more costly: false positives or false negatives?

#### Existing Infrastructure

* Are there existing ML models or datasets?
* Is there feedback from human moderators?

---

### 2. System Overview

An end-to-end content moderation pipeline for Roblox images:

```
User Uploads Image
       ↓
Content Moderation API Gateway
       ↓
1. Hash-based Check (PhotoDNA or perceptual hash)
       ↓
2. ML Model Inference (Nudity, violence, hate, etc.)
       ↓
3. Threshold Filter & Rule Engine
       ↓
4. Moderation Action (Allow / Block / Human Review)
       ↓
5. Feedback Logging (User reports, moderator input)
       ↓
6. Continuous Learning (Active learning, retraining)
```

---

### 3. Data Curation

#### Sources

* Flagged historical content from Roblox moderators
* Public datasets: NSFW, ImageNet, OpenImages
* Synthetic content (augmentation, inpainting)
* Screenshots from Roblox UGC

#### Label Schema

Multi-label classification:

| Label        | Examples                          |
| ------------ | --------------------------------- |
| Nudity       | Skin exposure, avatars in bikinis |
| Violence     | Weapons, blood, injury            |
| Hate Symbol  | Racist signs, swastikas           |
| Gore         | Severed limbs, intense visuals    |
| Safe Content | Landscapes, avatars, gameplay     |

#### Label Strategy

* Human annotation with guidelines
* Redundant labeling to ensure quality
* Consensus resolution on edge cases

---

### 4. Model Architecture

#### Backbone

* Pretrained: ViT, ConvNeXt, Swin Transformer, CLIP ViT-B/16
* Input: 224x224x3 RGB
* Normalized per ImageNet stats

#### Head

* Fully connected layer with Sigmoid activation
* One probability per class

#### Loss

* Multi-label Binary Cross Entropy (BCE)

#### Augmentations

* Random resize/crop, blur, JPEG compression
* Simulate adversarial image variants

---

### 5. Inference Pipeline

#### 1. Preprocessing

* Images are resized to the model's expected input size (e.g., 224x224) and normalized.
* Additional transforms (e.g., stripping overlays, format corrections) may be applied to reduce spoofing.

#### 2. Hash Filtering

Hash filtering is used as a fast and reliable way to detect known previously-flagged content before invoking any ML model. It uses two main approaches:

**Perceptual Hashing (e.g., pHash, dHash, aHash):**

* Converts an image into a short fixed-length binary hash (e.g., 64 bits) based on visual characteristics like structure or luminance gradients.
* Designed to detect visually similar images even with small changes (e.g., scaling, color shift, cropping).
* Comparison is done via Hamming distance between hashes, with a threshold determining a match.
* Easy to implement and scale, but more sensitive to complex distortions or adversarial noise.

**PhotoDNA:**

* Developed by Microsoft, it creates a robust and privacy-preserving signature of the image.
* Uses block-wise grayscale gradients and transforms (like DCT) to produce a fingerprint.
* Extremely resistant to tampering (e.g., resizing, heavy compression, watermarking).
* Typically used for detecting highly sensitive or illegal content like CSAM.
* Cannot be reversed into the original image, maintaining user privacy.
* Operates through controlled access (e.g., law enforcement vetted databases).

**Key Differences:**

| Feature              | Perceptual Hashing       | PhotoDNA                         |
| -------------------- | ------------------------ | -------------------------------- |
| Open source          | Yes                      | No (proprietary)                 |
| Use case flexibility | General image similarity | Specific to known illegal images |
| Robustness           | Moderate                 | Very High                        |
| Reversibility        | Not reversible           | Not reversible                   |
| Deployment model     | Local, lightweight       | Cloud-based, license-restricted  |

**Hash Database & Search Infrastructure:**
To support fast and scalable hash lookups:

* **Database Type:** Use an in-memory key-value store like **Redis**, **RocksDB**, or **LMDB** for fast exact match or near-match lookups.
* **For Approximate Nearest Neighbor (ANN):** Use systems like **FAISS**, **Annoy**, or **ScaNN** for efficient similarity search (Hamming or Euclidean distance).

**What is LSH (Locality Sensitive Hashing)?**

* LSH is a probabilistic method to hash high-dimensional data such that similar items map to the same or similar hash buckets.
* It allows **approximate nearest neighbor search** by ensuring that similar vectors are more likely to collide in the same hash bin.
* LSH is particularly useful when exact match is unnecessary or computationally expensive.
* For image hashes, LSH reduces search time from linear (O(N)) to sublinear (e.g., O(log N) or constant-time bucket lookups).

**What is Sharding?**

* Sharding is the process of **splitting a large dataset across multiple machines or partitions** to support scalability.
* In hash filtering, the hash index can be sharded based on categories (e.g., nudity, violence), image origin (e.g., country or region), or hash prefixes.
* Each shard handles a subset of the database, enabling **parallelism and low latency** during lookup.
* Sharding is critical for systems operating at large scale, where storing and querying all hashes on a single node is infeasible.

**Update Strategy:**

* Maintain a batch or stream-based update mechanism to ingest new hashes from human moderation or third-party blacklists.

Hash filtering reduces load on the model and enables fast rejection of re-uploaded toxic content.

#### 3. Model Selection

* A single general-purpose multi-label classifier model handles the majority of images.
* For performance and accuracy, a multi-stage architecture may be used:

  * Stage 1: Lightweight model (ResNet-18) to quickly eliminate clearly safe content.
  * Stage 2: Heavy ViT/CLIP-based model for uncertain or high-risk cases.
  * Optional branches: OCR module, face/pose detectors, avatar verification.

#### 4. Model Forward Pass

* The selected model outputs per-class probabilities for each moderation category (e.g., \[nudity: 0.93, violence: 0.12, hate: 0.01]).
* Confidence thresholds are dynamically tunable (e.g., allow, review, block) and may be class-specific.

#### 5. Rule-Based Filters

* A business logic engine evaluates additional non-ML rules:

  * **Whitelisting:** common, verified Roblox assets
  * **Avatar-Only Detection:** skip or lower thresholds for virtual humanoids vs real humans
  * **Overlay detectors:** if text/images obscure primary content
* Rule-based overrides allow for explainability and reduce over-reliance on black-box predictions.

#### 6. Decision Engine

* Based on ML outputs and rule-based filters:

  * If any class score exceeds a high-confidence threshold → **Auto-block**
  * If scores are in an uncertain range → **Queue for human review**
  * If all scores are low and rules pass → **Allow upload**

---

### 6. Human-in-the-Loop & Feedback

* UI for moderators to confirm or override decisions
* User reports feed into flagged queue
* Store moderator actions as ground truth

#### Feedback Loop

* Periodic retraining from false positives/negatives
* Active learning to mine uncertain or novel samples

---

### 7. Deployment & Serving

#### Batching vs. Online Streaming

In the context of model serving, there are two main modes of processing image moderation requests:

**Batching:**

* The system collects multiple image inputs together into a batch before sending them through the model.
* Pros:

  * Higher throughput: GPUs process large batches efficiently.
  * Lower per-image compute cost.
* Cons:

  * Slight latency delay due to waiting for batch to fill.
  * Less responsive for individual image requests.
* Example: If 16 users upload images around the same time, the system can bundle these into a 16-image batch and process them in a single GPU call.

**Online Streaming (Real-time):**

* Each image is processed as soon as it arrives.
* Pros:

  * Low latency per request; better for real-time moderation.
  * No need to wait for batch fill.
* Cons:

  * Lower GPU utilization; less efficient than batching.
  * Can be costly at scale.
* Example: As soon as a single user uploads an image, it is sent immediately to the model and results are returned within milliseconds.

**Hybrid Approach:**

* Use small dynamic batch windows (e.g., wait 10ms and batch up to 8 requests) to balance latency and GPU efficiency.
* Online requests can fall back to default models if batch queue is busy.

The choice between batching and streaming depends on use case:

* **Streaming:** critical when moderation must happen in near-real-time (e.g., livestream avatars).
* **Batching:** ideal for post-upload scans, nightly audits, or low-priority moderation queues.

#### What is Kubernetes (Simplified Explanation)

Kubernetes is a system that helps you run and manage software applications in the cloud automatically. Think of it as a smart manager for your applications:

* It keeps your applications running all the time, even if some machines fail.
* It makes sure your app can scale — by starting more copies (called *pods*) when traffic increases.
* It organizes and distributes your app across multiple computers (called *nodes*) without you doing it manually.
* It automatically restarts crashed components and handles upgrades safely.

Why do we need it?

* **Scalability:** Traffic on Roblox can spike during peak hours. Kubernetes ensures the moderation system can handle more image requests by launching more instances.
* **Resilience:** If something crashes, Kubernetes restarts it automatically.
* **Deployment Automation:** You can roll out new versions of your model with zero downtime.
* **Resource Management:** It allocates GPU/CPU efficiently to inference workloads.

Example: When thousands of users upload images simultaneously, Kubernetes can launch more GPU-powered model servers to handle the load — and remove them later to save cost.

---

#### Serving Stack

* **Model Export:** Models are exported in ONNX or TorchScript format for compatibility and optimization.
* **Inference Host:** Use Triton Inference Server (NVIDIA) or TensorFlow Serving for high-throughput, multi-model serving.
* **Deployment Format:** Containerize the model server (e.g., Docker) and expose via REST/gRPC APIs.
* **Batching & Pipelining:** Enable dynamic batching for throughput and queue management. Pipeline preprocessing and postprocessing to share GPU memory.
* **Multi-model Routing:** If using lightweight and heavyweight cascades, use a router service to direct input to the appropriate model.

#### Load Balancer (What it is and Why it's Used)

A **load balancer** is a traffic manager that distributes incoming requests across multiple backend services or servers. It acts like a traffic cop at a busy intersection, making sure that no single server gets overwhelmed while others sit idle.

**Why do we use it?**

* **Distributes traffic** so that no server is overloaded
* **Improves reliability** by rerouting traffic if one server fails
* **Enables scaling** by adding or removing servers without downtime
* **Supports blue/green or canary deployments** by directing traffic to new model versions gradually

**Types:**

* **L4 Load Balancer:** Balances traffic based on transport layer (IP and port)
* **L7 Load Balancer:** Balances traffic based on HTTP-level info (like URL path, headers)

**In our system:**

* We use a Layer 7 (L7) load balancer (e.g., Envoy or Istio) to route image moderation requests to the appropriate Kubernetes pods.
* It can be configured to route based on path (e.g., `/v1/moderate`) or version headers for A/B testing.

---

#### Scaling Strategy

* **Horizontal Autoscaling:** Use Kubernetes HPA (Horizontal Pod Autoscaler) to adjust the number of inference pods based on CPU/GPU load or request volume.
* **GPU Scheduling:** Schedule heavy models on GPU-enabled nodes; reserve CPUs for lightweight or hash-filter-only traffic.
* **Load Balancer:** Use an L7 load balancer (e.g., Envoy, Istio) for traffic routing and TLS termination.
* **Cold Start Optimization:** Pre-warm model instances to avoid cold start delays.

#### Monitoring & Observability

* **Latency Monitoring:** Use tools like Prometheus + Grafana to monitor inference latency distribution.
* **Request Tracing:** Enable distributed tracing (e.g., OpenTelemetry, Jaeger) to debug bottlenecks.
* **Model Drift Detection:** Compare incoming embeddings or class distributions over time against training baseline to detect drift.
* **Alerting:** Define SLA thresholds for max latency, throughput drop, or accuracy degradation and trigger alerts.

#### A/B and Canary Deployment

**Canary Deployment:**

* A canary deployment gradually rolls out a new model version to a **small percentage of live traffic** (e.g., 1–5%) while keeping the old version active.
* Purpose: detect regressions or performance issues early.
* If the new model performs well, traffic is increased incrementally until full adoption.
* If issues arise, traffic is quickly rolled back to the old model.
* Canary testing is often used for production **stability and safety checks**.

**A/B Testing:**

* A/B testing splits traffic into **distinct segments** (e.g., 50/50) where group A sees version A of the model and group B sees version B.
* Purpose: compare performance of different models or policies under identical real-world conditions.
* Metrics (e.g., accuracy, moderation agreement, user appeal rate) are tracked independently for each group.
* A/B testing is often used to **evaluate model impact** rather than detect breakage.

**Key Differences:**

| Feature            | Canary Deployment            | A/B Testing                        |
| ------------------ | ---------------------------- | ---------------------------------- |
| Goal               | Stability, safety            | Comparison, experimentation        |
| Traffic allocation | Gradual ramp-up              | Fixed split between variants       |
| Rollback friendly  | Yes                          | No (used for analysis only)        |
| Metrics            | Monitor regressions          | Analyze effectiveness              |
| Usage context      | Production deployment checks | Model evaluation & experimentation |

* Roll out new model versions to a subset of traffic (e.g., 5%) for evaluation.
* Monitor KPIs during rollout before full switch-over.
* Canary strategy helps detect regressions early and allows rollback if needed.

#### Deployment Workflow

1. Train → validate model locally
2. Export to ONNX/TorchScript
3. Containerize with preprocessing and postprocessing logic
4. Push to image registry (e.g., ECR, GCR)
5. Deploy to Kubernetes using Helm or ArgoCD
6. Register service in service mesh
7. Monitor health, latency, throughput, and accuracy

---

### 8. Evaluation Metrics

#### What is ROC-AUC-Off-line?

**ROC-AUC** stands for **Receiver Operating Characteristic - Area Under the Curve**. It is a performance measurement for classification problems at various threshold settings.

* **ROC Curve:** Plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at different thresholds.
* **AUC (Area Under the Curve):** Measures the entire two-dimensional area under the entire ROC curve. The value is between 0 and 1.

**Why is it useful?**

* Unlike accuracy, ROC-AUC evaluates how well the model ranks positive samples higher than negatives regardless of class imbalance.
* An AUC of 0.5 means the model is no better than random guessing.
* An AUC of 1.0 means perfect classification.

**Why is ROC-AUC more stable toward class imbalance?**

* Metrics like accuracy or precision can be misleading when most data points belong to the negative class (e.g., 98% safe images and 2% harmful).
* A model predicting all samples as negative would still get 98% accuracy — but it's useless.
* ROC-AUC, however, looks at the model’s ability to rank **a randomly chosen positive sample higher than a randomly chosen negative one**.
* It doesn’t depend on the absolute counts of positive or negative samples, so it gives a fairer view of discriminative power.
* Thus, it's more stable and reliable when your dataset is highly skewed, which is common in safety tasks like image moderation.

**In image moderation:**

* ROC-AUC helps assess how well the model separates toxic and safe content, especially when the number of positive examples (e.g., offensive content) is much smaller than negatives.

* It’s often more stable than precision or recall alone across different decision thresholds.
  **ROC-AUC** stands for **Receiver Operating Characteristic - Area Under the Curve**. It is a performance measurement for classification problems at various threshold settings.

* **ROC Curve:** Plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at different thresholds.

* **AUC (Area Under the Curve):** Measures the entire two-dimensional area under the entire ROC curve. The value is between 0 and 1.

**Why is it useful?**

* Unlike accuracy, ROC-AUC evaluates how well the model ranks positive samples higher than negatives regardless of class imbalance.
* An AUC of 0.5 means the model is no better than random guessing.
* An AUC of 1.0 means perfect classification.

**In image moderation:**

* ROC-AUC helps assess how well the model separates toxic and safe content, especially when the number of positive examples (e.g., offensive content) is much smaller than negatives.
* It’s often more stable than precision or recall alone across different decision thresholds.

| Metric               | Purpose                           |
| -------------------- | --------------------------------- |
| ROC-AUC              | Handles class imbalance           |
| Precision\@Threshold | Avoid blocking safe content       |
| Recall\@Threshold    | Ensure most violations are caught |
| Inference Latency    | Stay within real-time constraints |
| Model Size           | Supports scaling cost estimation  |

| Metric               | Purpose                           |
| -------------------- | --------------------------------- |
| ROC-AUC              | Handles class imbalance           |
| Precision\@Threshold | Avoid blocking safe content       |
| Recall\@Threshold    | Ensure most violations are caught |
| Inference Latency    | Stay within real-time constraints |
| Model Size           | Supports scaling cost estimation  |

### Online KPIs:

$$
\begin{align}
\text{Apeals} &=& \frac{\text{Number of reversed apeals}}{\text{Number of harmful posts detected by the system}}\\
\text{Proactive Rate}&=&\frac{\text{Number of harmful posts detected by the system}}{\text{Number of harmful posts detected by the system + reposted by users}}
\end{align}
$$

---

### 8.5 Handling Class Imbalance in Training

Moderation datasets are often highly imbalanced — for example, the majority of images may be safe, while only a small portion contain harmful content like nudity or hate symbols. Training on such data without adjustments can lead to poor model performance on rare but important classes. Several strategies can help:

**1. Class Weighting in Loss Function**

* Assign higher loss weights to underrepresented classes.
* Example: In Binary Cross Entropy Loss, pass a `weight` vector to give more importance to rare labels.

**2. Focal Loss**

* A modified loss function that down-weights easy examples and focuses training on hard or minority examples.
* Helps prevent model from being overwhelmed by frequent negative labels.

**3. Data Resampling**

* **Oversampling:** Duplicate examples of minority classes.
* **Undersampling:** Reduce the number of majority class examples.
* **Hybrid:** Combine both methods to balance classes.
* Note: Risk of overfitting with oversampling.

**4. Synthetic Data Augmentation**

* Use techniques like image transformation, generative models, or inpainting to create new examples of rare classes.
* Helps increase variation and robustness.

**5. Curriculum or Hard Example Mining**

* Focus training more on examples that the model currently misclassifies, especially in minority classes.
* Use confidence thresholds or margin-based sampling to identify hard negatives.

**6. Multi-Task Learning**

* Combine related labels (e.g., NSFW, violence, suggestive) in shared representation.
* Leverages data across related tasks to improve rare class performance.

These strategies ensure that the model doesn’t simply learn to predict the majority (safe) class and helps it detect harmful content reliably — a key requirement for trust and safety systems.

---

### 8.6 Handling Adversarial Image Perturbations

Adversarial perturbations are small, often imperceptible changes made to images to intentionally fool machine learning models into making incorrect predictions. In image moderation, these can be used to bypass automated filters and upload unsafe content.

**Common Perturbation Techniques:**

* Adding slight noise to critical regions
* Overlaying semi-transparent textures or stickers
* Embedding nudity/violence into busy backgrounds
* Using GANs or inpainting to mask offensive areas

**Defenses and Mitigations:**

**1. Robust Data Augmentation**

* During training, simulate perturbations using noise, JPEG compression, blurring, occlusion, and overlays.
* Improves the model's ability to generalize and handle subtle attacks.

**2. Input Normalization and Sanitization**

* Apply image prefilters (e.g., denoising, contrast normalization) before inference.
* Remove metadata or overlay layers that can be exploited.

**3. Ensemble or Cascaded Models**

* Use multiple models or stages to make predictions.
* A perturbation that fools one model is less likely to fool all.

**4. Gradient Masking or Adversarial Training**

* **Adversarial Training:**

  * During training, generate adversarial examples using methods like FGSM (Fast Gradient Sign Method) or PGD (Projected Gradient Descent).
  * These examples are added to the training set, teaching the model to correctly classify perturbed inputs.
  * Helps the model learn more robust decision boundaries that are harder to exploit.

* **Gradient Masking:**

  * A defensive technique where the model suppresses or obfuscates its gradient information.
  * Makes it difficult for attackers to compute meaningful gradients for crafting adversarial examples.
  * However, this can be fragile: it may provide a false sense of security and break under stronger white-box attacks.
  * Generally, adversarial training is considered more principled and effective than relying on masking alone.

**5. Human-in-the-loop Review for Low-Confidence Predictions**

* For images near decision boundaries or exhibiting visual anomalies, route to a moderator.
* This mitigates risk when model confidence drops due to adversarial interference.

**6. Detection Heuristics**

* Train a separate model or rule-based filter to detect adversarial patterns (e.g., checkerboard noise, unnatural textures).
* Use similarity scores to check if the image deviates too far from expected visual distributions.

Together, these strategies increase the resilience of the moderation pipeline and reduce the risk of model evasion in adversarial settings.

Adversarial perturbations are small, often imperceptible changes made to images to intentionally fool machine learning models into making incorrect predictions. In image moderation, these can be used to bypass automated filters and upload unsafe content.

**Common Perturbation Techniques:**

* Adding slight noise to critical regions
* Overlaying semi-transparent textures or stickers
* Embedding nudity/violence into busy backgrounds
* Using GANs or inpainting to mask offensive areas

**Defenses and Mitigations:**

**1. Robust Data Augmentation**

* During training, simulate perturbations using noise, JPEG compression, blurring, occlusion, and overlays.
* Improves the model's ability to generalize and handle subtle attacks.

**2. Input Normalization and Sanitization**

* Apply image prefilters (e.g., denoising, contrast normalization) before inference.
* Remove metadata or overlay layers that can be exploited.

**3. Ensemble or Cascaded Models**

* Use multiple models or stages to make predictions.
* A perturbation that fools one model is less likely to fool all.

**4. Gradient Masking or Adversarial Training**

* Incorporate adversarial examples into training using methods like FGSM or PGD.
* The model learns to become insensitive to adversarial changes.

**5. Human-in-the-loop Review for Low-Confidence Predictions**

* For images near decision boundaries or exhibiting visual anomalies, route to a moderator.
* This mitigates risk when model confidence drops due to adversarial interference.

**6. Detection Heuristics**

* Train a separate model or rule-based filter to detect adversarial patterns (e.g., checkerboard noise, unnatural textures).
* Use similarity scores to check if the image deviates too far from expected visual distributions.

Together, these strategies increase the resilience of the moderation pipeline and reduce the risk of model evasion in adversarial settings.

---

### 9. Continuous Learning Strategy

* Daily logs analyzed for misclassifications
* Use hard examples in next round of fine-tuning
* Mix in synthetic edge cases
* Retrain monthly or as needed

---

### 10. Discussion Points for Interview

* Tradeoff between latency and accuracy
* Auto-ban vs human-in-the-loop threshold setting
* Handling adversarial image perturbations
* Role of OCR and text detection in image moderation
* Risks of bias (e.g. cultural perceptions of nudity)
* Multi-modal extension: combine with chat or metadata

---

This system is modular, scalable, and suitable for Roblox's unique needs in moderating user-generated image content with safety and performance guarantees.
