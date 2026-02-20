# DHAI-4: Deep Hierarchical Active Inference (HDC Pivot)

DHAI-4 is an experimental Artificial General Intelligence (AGI) architecture that completely abandons connectionist neural networks (e.g., Transformers, Backpropagation). Instead, it implements **Language-Mediated Active Inference** using the pure, deterministic linear algebra of **Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA)**.

This repository demonstrates how to build a model that natively grows semantic and syntactic structures from streaming raw Wikipedia text without a single learned matrix weight, scalar parameter explosion, or GPU optimization loop.

---

## 1. The Core Problem: The "Black Box" & Parameter Explosion
Modern Large Language Models (LLMs) learn by iteratively adjusting continuous gradients over billions of parameters, resulting in an uninterpretable "black box" latent space. 

Earlier iterations of DHAI attempted to solve this by forcing the model to use purely discrete, observable integer states (classic Active Inference). However, representing the vast combinatorial scope of human language using strictly discrete integers leads to an $\mathcal{O}(N^2)$ combinatorial explosion in the transition matrices, causing the models to fail at generalizing past initial training sets (a failure state known as "Fluent Incoherence").

DHAI-4 solves this by pivoting the discrete mathematical substrate into a high-dimensional geometric space.

---

## 2. Foundational Mathematics: Native Semantic Geometry

Hyperdimensional Computing operates on the mathematical principle of concentration of measure in vastly high-dimensional spaces. We construct a vector space of dimension $d = 10,000$. 

The model relies on vectors whose components are strictly bipolar:
$$ V \in \{-1, +1\}^d $$

Because $d$ is immense, any two randomly generated vectors are overwhelmingly likely to be mathematically orthogonal (dissimilar). The core mathematics does not rely on optimization, but on four deterministic algebraic operations: Generation, Binding, Bundling, and Permutation.

### 2.1 The Vocabulary: Atomic Generation
When the system encounters a novel concept (e.g., the word "Cat"), it does *not* train an embedding using Word2Vec or Transformers. It simply generates a completely random $10,000$-dimensional vector of $+1$s and $-1$s.

$$ V_{\text{cat}} \sim \mathcal{U}\left(\{-1, +1\}^{10000}\right) $$
$$ V_{\text{dog}} \sim \mathcal{U}\left(\{-1, +1\}^{10000}\right) $$

We measure distance using **Cosine Similarity**, which in a bipolar space is an exact, computationally cheap translation of the Hamming Distance:

$$ \text{Sim}(V_A, V_B) = \frac{V_A \cdot V_B}{d} $$

Because the space is random and $d=10,000$, the expected dot product of any two random vectors evaluates to approximately zero.
$$ \text{Sim}(V_{\text{cat}}, V_{\text{dog}}) \approx 0 $$

### 2.2 Structural Factorization: Binding ($\otimes$)
To learn grammatical rules without parameter explosion (The "Chomsky Fix"), we must decouple the *structural role* of a word from its *semantic meaning*. We achieve this via the **Binding** operator. 

Binding acts as variable assignment: $\text{Role} \times \text{Filler}$. Mathematically, this is the element-wise multiplication of the two vectors (computationally equivalent to an `XOR` gate on binary hardware).

$$ V_{\text{bound}} = V_{\text{role\_subject}} \otimes V_{\text{filler\_cat}} $$

**The Crucial Geometric Property**: The resulting bound vector $V_{\text{bound}}$ is perfectly orthogonal to *both* of its parent vectors.
$$ \text{Sim}(V_{\text{bound}}, V_{\text{role\_subject}}) \approx 0 \quad \text{and} \quad \text{Sim}(V_{\text{bound}}, V_{\text{filler\_cat}}) \approx 0 $$
This creates a fundamentally unique state in the state-space that perfectly encodes structure, without increasing the overall dimensionality of the system.

### 2.3 Semantic Clustering: Bundling ($+$)
To aggregate multiple concepts into macroscopic "Events" or sets, we use the **Bundling** operator. This is the element-wise addition of multiple vectors, followed by a non-linear thresholding function (the signum function) to compress the magnitude back into the strict bipolar $\{-1, +1\}$ space.

$$ S = V_A + V_B + V_C $$
$$ V_{\text{bundle}} = \text{sgn}(S) $$
*(Where $\text{sgn}(0)$ is resolved via a random coin flip to break ties).*

**The Crucial Geometric Property**: Unlike Binding, a Bundled vector is geometrically highly similar to all of its constituent input vectors.
$$ \text{Sim}(V_{\text{bundle}}, V_A) \gg 0 $$

This mathematical property produces **Zero-Shot Generalization**. If the model records a transition rule for $V_{\text{bundle}}$, and later evaluates a novel context that is geometrically similar to the bundle, the dot product will naturally fire and apply the learned rules without ever executing backpropagation.

### 2.4 Sequence and Time: Permutation ($\rho$)
To differentiate sequences (e.g., distinguishing "The Cat" from "Cat The"), we apply a **Permutation** operator. The simplest functioning permutation is a cyclical shift of the vector coordinates by 1 offset.

$$ V_{\text{time}} = \rho(V_{\text{The}}) $$
$$ \text{Sim}(\rho(V_{\text{The}}), V_{\text{The}}) \approx 0 $$
By cyclically shifting vectors before binding or bundling them, we structurally embed the concept of Time into the static geometry.

---

## 3. The Architecture: Hierarchical VSA Processing

The DHAI-4 repository maps the HDC algebra directly into a simulated 3-layer brain hierarchy.

### Level 0: The Sensory Interface (Broca's Area)
**File**: `models/level0_broca.py`
- **Mechanism**: Raw text strings are ingested. Words are converted to atomic vectors and **Bound** ($\otimes$) to their syntactic Roles. 
- **Learning**: The system bypasses gradient descent in favor of **Geometric Hebbian Learning**. When state $A$ transitions to state $B$, the system just records $\text{count}(A \to B) \mathrel{+}= 1$. Native geometric similarity handles the generalization mapping during inference.

### Level 1: The Event Processor (Wernicke's Area)
**File**: `models/level1_wernicke.py`
- **Mechanism (Fast Structure Learning)**: The system computes continuous **Bundles** ($+$) of the incoming token stream. 
- **Chunking**: It actively monitors the Cosine Similarity between the newest token and the current rolling Bundle. If $\text{Sim}(V_{\text{new}}, V_{\text{context}}) < \text{Threshold}$, the architecture assumes a linguistic boundary has been crossed, finalizes the macroscopic "Event", and starts a new bundle.

### Level 2: The Narrative Planner (Frontal Cortex)
**File**: `models/level2_frontal.py`
- **Mechanism (Sophisticated Inference)**: Active Inference requires the model to plan actions that minimize Variational Free Energy. 
- **Deep Tree Search**: The planner does not blindly guess the next token. It computes an $N$-step recursive tree search. For every candidate action, it simulates the `Virtual Future = Bundle(Current Belief, Proposed Action)`.
- **Expected Free Energy (EFE)**: EFE is computed as the strict Cosine Distance from the `Virtual Future` vector to the `Goal` vector over the simulated horizon.

---

## 4. Addressing Mathematical Limits: Sleep Consolidation

Vector Symbolic Architectures are subject to the **Superposition Catastrophe**: bundling too many vectors together eventually collapses the vector into orthogonal noise, destroying the capacity to retrieve constituent elements.

To counteract this, Level 0 utilizes **Bayesian Model Reduction (Sleep Phase)**. 
During offline intervals between training batches, the `sleep_cycle()` method iterates over the Hebbian transition graph and enforces sparsity. It prunes weak connections iteratively, orthogonalizing the semantic representations to reclaim memory capacity mathematically without needing additional hard drive storage.

---

## 5. Usage: Infinite Modern English Streaming

The repository features an infinite streaming trainer that queries the Wikipedia API. It pulls random articles, processes them through Levels 0, 1, and 2 to build the geometric graphs, and instantly discards the raw text.

To run the pipeline and watch the model build natively structured semantic spaces:

```bash
# 1. Clone the repository
git clone https://github.com/RatnaAnimesh/hai-txt.git
cd hai-txt

# 2. Run the infinite streaming loop
python dhai4_hdc/train_modern_english.py
```
