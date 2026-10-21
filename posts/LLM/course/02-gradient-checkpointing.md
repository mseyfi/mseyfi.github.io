# Lesson 02: Gradient Checkpointing

## The main idea

During training, the forward pass computes activations. Backpropagation needs some of these values to calculate gradients. Saving many activations uses GPU memory.

**Gradient checkpointing** (also called *activation checkpointing*) saves selected boundary tensors and recomputes other activations during the backward pass. It trades extra computation for lower activation memory.

[Explore the editable DAG on tldraw](https://www.tldraw.com/f/rZ3eUXC6bkMSm0eFbzVCz).

## A tiny example

Let the input be \(x=3\), the weight be \(w=2\), and the target be \(t=10\).

**Forward:**

\[
y=wx=2\times3=6
\]

\[
L=(y-t)^2=(6-10)^2=16
\]

**Backward (chain rule):**

\[
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial y}\frac{\partial y}{\partial w}
=2(y-t)x
=2(6-10)\times3=-24
\]

Notice the factor \(x\): computing the weight gradient requires the input value (or an equivalent way to recover it). Autograd saves tensors needed by each backward operation; it does **not** necessarily save every layer output.

## Four-layer network

\[
a_0=x,\qquad a_i=f_i(a_{i-1};\theta_i),\quad i=1,2,3,4
\]

Without checkpointing, the framework may retain intermediate tensors for backward.

With checkpointing, imagine saving the input \(a_0\) and the boundary activation \(a_2\). Activations inside the checkpointed regions need not all remain stored. During backward, the framework reruns the required forward operations from a saved boundary:

\[
a_3=f_3(a_2;\theta_3),\qquad a_4=f_4(a_3;\theta_4)
\]

Then it uses the reconstructed values to calculate gradients.

## The trade-off

\[
\boxed{\text{Less activation memory}\quad\Longleftrightarrow\quad\text{More forward computation}}
\]

Checkpointing reduces **activation memory**. It does not directly reduce memory used by model parameters, gradients, or optimizer states. Actual savings and speed cost depend on checkpoint boundaries and model architecture.

## Where should we place checkpoints?

Checkpoint boundaries determine which tensors stay in memory and which computations must be repeated. For a sequential network, grouping several layers into each checkpointed region is a common starting point. There is no single optimal spacing for every model: test peak GPU memory and training step time.

## Check your understanding

If we saved \(a_2\), can we reconstruct \(a_3\)? Yes: run layer 3 again with \(a_2\).

If training is slower with checkpointing, why use it? Because the model or batch may otherwise not fit in GPU memory.

## PyTorch example

```python
import torch
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.relu(x @ W)

# W is a trainable tensor.
# Recompute the block during backward instead of retaining
# all tensors that normal autograd would save inside it.
y = checkpoint(block, x, use_reentrant=False)
```

**Note:** The code is illustrative; define `W` and `x` with suitable shapes and gradient settings before running it. Random operations and stateful behavior need care to preserve correct recomputation.
