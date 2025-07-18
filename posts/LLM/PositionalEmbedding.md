# Why positionnal Embedding Matters?

Positional embeddings are crucial for transformers. Without them, the model is fundamentally **permutation invariant**, meaning it treats the input as a "bag of words" rather than an ordered sequence.

You're right that if the input is reordered, the attention scores are also reordered. However, the model still sees the inputs as fundamentally the same because the relationships it calculates are based purely on the content of the words, not their positions.

Here's a breakdown of why this happens:

### The "Bag of Words" Problem in Self-Attention

At its core, the self-attention mechanism in a transformer calculates a score for every word in the input sequence against every other word. This score determines how much "attention" a word should pay to another when creating its updated representation.

Let's simplify the process for the word "cat" in both your example sentences, assuming we have no positional information:

* **Sentence 1:** "the", "cat", "chased", "the", "dog"
* **Sentence 2:** "the", "dog", "chased", "the", "cat"

The transformer first creates Query (Q), Key (K), and Value (V) vectors for each word. These are learned representations of the words themselves. Crucially, without positional embeddings, the vectors for "cat" are identical in both sentences. The same goes for "dog," "chased," and "the."

Now, to calculate the attention score for "cat," its Query vector is multiplied by the Key vector of every other word in the sentence:

* **For "cat" in Sentence 1:**
    * Attention("cat", "the")
    * Attention("cat", "cat")
    * Attention("cat", "chased")
    * Attention("cat", "the")
    * Attention("cat", "dog")

* **For "cat" in Sentence 2:**
    * Attention("cat", "the")
    * Attention("cat", "dog")
    * Attention("cat", "chased")
    * Attention("cat", "the")
    * Attention("cat", "cat")

Notice that the set of attention scores calculated for "cat" is **exactly the same** in both sentences, just in a different order. The final representation for "cat" is a weighted sum of the Value vectors of all words in the sentence, where the weights are these attention scores. Since the set of words and their corresponding scores are identical, the resulting weighted sum is also identical.

The feed-forward network that follows the attention layer receives this context-rich representation of "cat." However, since this representation is the same for both sentences, the network has no way of knowing that the surrounding words were in a different order.

### The Role of Positional Embeddings

This is where positional embeddings come in. They are vectors that represent the position of a word in a sequence. This positional information is added to the initial word embeddings.



Now, the initial vector for "cat" at position 1 in the first sentence is different from the vector for "cat" at position 4 in the second sentence. This difference propagates through the entire process:

* The Query, Key, and Value vectors for "cat" will be different in each sentence.
* The attention scores will be different because they are calculated using these position-aware vectors.
* The final representation of "cat" will be different, reflecting its unique context in each sentence.

In essence, by adding a unique "signature" for each position, you give the model the necessary information to understand the order of the words and, therefore, the different meanings of "the cat chased the dog" and "the dog chased the cat."

Of course. We can prove mathematically that for a self-attention layer without positional embeddings, permuting the input sequence results in an identical permutation of the output sequence. This property is known as **permutation equivariance**.

Here’s the step-by-step proof.

***

### Setup and Definitions

Let's define our terms:

* **Input Sequence**: Let the input be a sequence of $n$ token embeddings. We can represent this as a matrix $X$ of size $(n \times d_{model})$, where $n$ is the sequence length and $d_{model}$ is the dimension of each embedding vector.
    $$X = \begin{pmatrix} \text{— } x_1 \text{ —} \\ \text{— } x_2 \text{ —} \\ \vdots \\ \text{— } x_n \text{ —} \end{pmatrix}$$
* **Permutation**: A permutation of the input sequence can be represented by left-multiplying $X$ by a **permutation matrix** $P$. A permutation matrix is a square matrix with exactly one '1' in each row and column and '0's elsewhere.
    * The permuted input is $X^\prime = PX$.
* **Goal**: We want to prove that the output $Z^\prime$ (from the permuted input $X^\prime$) is equal to $PZ$, where $Z$ is the output from the original input $X$. This would mean the output is permuted in the exact same way as the input.
    $$Z^\prime = PZ$$
* **Self-Attention Formula**: The output $Z$ is calculated as:
    $$Z = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    Where $Q = XW_Q$, $K = XW_K$, and $V = XW_V$ are the Query, Key, and Value matrices, and $W_Q, W_K, W_V$ are the learned weight matrices.

### The Proof

Let's calculate the new output $Z^\prime$ using the permuted input $X^\prime = PX$.

**1. Calculate the new Query, Key, and Value Matrices ($Q^\prime, K^\prime, V^\prime$)**

We simply substitute $X^\prime$ for $X$ in the equations:
* $Q^\prime = X^\primeW_Q = (PX)W_Q = P(XW_Q) = PQ$
* $K^\prime = X^\primeW_K = (PX)W_K = P(XW_K) = PK$
* $V^\prime = X^\primeW_V = (PX)W_V = P(XW_V) = PV$

So, the new Q, K, and V matrices are just permuted versions of the original ones.

**2. Calculate the new Attention Score Matrix ($A^\prime$)**

The attention score matrix, $A^\prime$, is calculated using $Q^\prime$ and $K^\prime$:
$$A^\prime = \text{softmax}\left(\frac{Q^\primeK^\prime^T}{\sqrt{d_k}}\right)$$
Now, substitute the expressions for $Q^\prime$ and $K^\prime$:
$$A^\prime = \text{softmax}\left(\frac{(PQ)(PK)^T}{\sqrt{d_k}}\right)$$
Using the transpose property $(AB)^T = B^T A^T$, we get $(PK)^T = K^T P^T$:
$$A^\prime = \text{softmax}\left(\frac{P Q K^T P^T}{\sqrt{d_k}}\right)$$
The softmax function is applied row-wise. It can be shown that applying a permutation to the rows (via $P$) and columns (via $P^T$) of a matrix *before* the softmax is equivalent to applying the permutation *after* the softmax. In other words, $\text{softmax}(PSP^T) = P(\text{softmax}(S))P^T$.
Letting $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$, we get:
$$A^\prime = P A P^T$$

**3. Calculate the Final Output ($Z^\prime$)**

The final output is the product of the attention scores and the Value matrix:
$$Z^\prime = A^\primeV^\prime$$
Now, substitute the expressions we found for $A^\prime$ and $V^\prime$:
$$Z^\prime = (PAP^T)(PV)$$
A key property of any permutation matrix $P$ is that its transpose is its inverse, meaning $P^T P = I$, where $I$ is the identity matrix.
$$Z^\prime = P A (P^T P) V$$
$$Z^\prime = P A (I) V$$
$$Z^\prime = P A V$$
Since the original output was $Z = AV$, we can substitute that back in:
$$Z^\prime = P Z$$

### Conclusion

We have successfully shown that $Z^\prime = PZ$. This proves that if you permute the input sequence ($X^\prime = PX$), the output of the self-attention layer is the original output with the exact same permutation applied to it ($Z^\prime = PZ$).

The model doesn't see the reordered inputs as "the same" in the sense of producing an identical, un-shuffled output. Instead, it processes them in a way that is perfectly symmetrical to their order. Because the output order is just a reflection of the input order and not a result of learning the *meaning* of that order, the transformer requires positional embeddings to break this symmetry and understand sequential information.
