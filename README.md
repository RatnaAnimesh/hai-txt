# DHAI-4: Deep Hierarchical Active Inference via Hyperdimensional Computing

**Abstract:** *This paper introduces the theoretical and architectural foundations of DHAI-4 (Deep Hierarchical Active Inference 4), an Artificial General Intelligence (AGI) framework designed to perform language-mediated Sophisticated Inference. We detail the mathematical pivot from continuous connectionist models (e.g., Transformers, Backpropagation) and discrete integer-based Partially Observable Markov Decision Processes (POMDPs) into the strict deterministic linear algebra of Vector Symbolic Architectures (VSAs) operating in a $d=10,000$ geometric space. We demonstrate how this mapping naturally supports native syntactic factorization, zero-shot topological generalization, and mathematically tractable calculations of Expected Free Energy without parameter explosion or "black-box" optimization loops.*

---

## 1. Introduction: The Free Energy Principle

The core of the DHAI framework relies on the **Free Energy Principle (FEP)**, which posits that self-organizing systems must minimize Variational Free Energy (an upper bound on surprise) to resist entropic decay. 

In a discrete time process, an agent exists in an environment governed by hidden states $s$, producing observations $o$. Formally, this is modeled as a POMDP defined by:

- $P(o_\tau | s_\tau)$: The likelihood matrix (Generation).
- $P(s_{\tau} | s_{\tau-1}, \pi)$: The transition matrix (Dynamics under policy $\pi$).
- $P(s_0)$: Prior belief over initial states.

The agent maintains an approximate posterior belief $Q(s)$ over hidden states. Perception and action are cast as a single imperative: minimizing **Variational Free Energy ($F$)**, bounding the surprise $-\ln P(o)$.

$$
F = D_{KL}[Q(s) \parallel P(s)] - \mathbb{E}_{Q}[\ln P(o|s)]
$$

Where $D_{KL}$ is the Kullback-Leibler divergence. For planning into the future ($\tau > t$), the agent evaluates policies ($\pi$) by minimizing **Expected Free Energy ($G$)**:

$$
\begin{aligned}
G(\pi, \tau) &= \mathbb{E}_{\tilde{Q}} [ \ln Q(s_\tau | \pi) - \ln P(s_\tau, o_\tau | \pi) ] \\
&\approx \underbrace{- \mathbb{E}_{\tilde{Q}} [\ln P(o_\tau)]}_{\text{Pragmatic Value}} + \underbrace{\mathbb{E}_{\tilde{Q}} [H(P(o_\tau | s_\tau))]}_{\text{Epistemic Value}}
\end{aligned}
$$

---

## 2. The Combinatorial Catastrophe of Discrete POMDPs

Attempting to scale classic Active Inference to the domain of natural language fundamentally breaks because integer-based POMDP matrices suffer $\mathcal{O}(|S|^2)$ or worse dimensional scaling.

If a language agent has a vocabulary of $|V| = 50,000$ words:

1. **State Space Explosion**: Defining states as strictly discrete integers means the transition matrix $P(s_t | s_{t-1})$ requires $|V| \times |V| = 2.5 \times 10^9$ discrete scalar probabilities.
2. **Lack of Topological Similarity**: Discrete integers possess no innate geometry (state $1$ "Dog" and state $2$ "Cat" are as mathematically distant as state $1$ and state $50,000$ "Carburetor"). To generalize, the model must explicitly observe and iterate a continuous probability gradient linking them.

This leads to "Fluent Incoherence," where models memorize local transition probabilities but immediately fail at compositional generalization. The solution is escaping integer Markov chains by mapping the state space into a dense geometry.

---

## 3. Hyperdimensional Computing (HDC) & VSA Algebra

To resolve the combinatorial explosion natively, DHAI-4 maps the generative model into a massive Vector Symbolic Architecture (VSA). 

We define a vector space of dimension $d = 10,000$. The substrate consists of bipolar vectors:

$$
V \in \{-1, +1\}^d
$$

### 3.1 Concentration of Measure

In vastly high-dimensional spaces, the volume of a hypersphere concentrates heavily near its equator relative to any arbitrary pole. Consequently, any two randomly generated bipolar vectors are mathematically guaranteed to be orthogonal (dissimilar).

Let $x, y \sim \mathcal{U}(\{-1, +1\}^d)$. The Cosine Similarity is equivalent to the normalized Dot Product:

$$
\text{Sim}(x, y) = \frac{x \cdot y}{d} = \frac{1}{d} \sum_{i=1}^d x_i y_i
$$

By the Law of Large Numbers, $\mathbb{E}[\text{Sim}(x, y)] = 0$.

### 3.2 The Algebraic Operations

DHAI-4 uses three deterministic operations over this geometry.

**1. Generation / Item Memory:** 

An atomic concept (e.g., a word) is assigned a pure random vector.

$$
V_{\text{dog}} = \text{generate}(\{-1, +1\}^d)
$$

**2. Binding ($\otimes$):**

Element-wise multiplication (Hadamard product). Used for Structural Factorization (variable assignment).

$$
V_{\text{bound}} = x \otimes y
$$

*Geometrical Property:* $V_{\text{bound}}$ is perfectly orthogonal to both $x$ and $y$. 

$$
\text{Sim}(x \otimes y, x) \approx 0
$$

**3. Bundling ($+$):**

Element-wise addition followed by a signum threshold function. Used to create macroscopic Sets or semantic contexts.

$$
\begin{aligned}
S &= x + y + z \\
V_{\text{bundled}} &= \text{sgn}(S)
\end{aligned}
$$

*(Where $\text{sgn}(0)$ is resolved via a coin flip).*
*Geometrical Property:* The bundle retains high similarity to all constituent vectors.

$$
\text{Sim}(V_{\text{bundled}}, x) \gg 0
$$

**4. Permutation ($\rho$):**

A cyclical coordinate shift by 1 position. Used to encode sequence without commutativity.

$$
V_{\text{seq}} = \rho(x)
$$

*Geometrical Property:* $\text{Sim}(\rho(x), x) \approx 0$.

---

## 4. Architectural Implementation in DHAI-4

We construct a 3-tier hierarchy that maps active inference onto the VSA algebra.

### Level 0: The Sensory Interface (Broca / Syntactic Binding)

Modern LLMs conflate syntax and semantics into the same continuous embeddings. Level 0 strictly enforces **Structural Factorization**.

We define permanent, orthogonal functional roles: $R_{\text{SUBJ}}, R_{\text{VERB}}, R_{\text{OBJ}}$. 
When the system senses a word vector $F_{\text{cat}}$ (the Filler), it binds it to the current role:

$$
S_0 = R_{\text{SUBJ}} \otimes F_{\text{cat}}
$$

**Transition Updating:**

Instead of a $|V| \times |V|$ dense Markov matrix, transitioning from state $S_{\tau-1}$ to $S_{\tau}$ is tracked via a sparse Hebbian graph mapping geometric neighborhoods.

$$
\mathbf{\Gamma}(S_{\tau-1} \to S_{\tau}) \leftarrow \mathbf{\Gamma}(S_{\tau-1} \to S_{\tau}) + 1
$$

Because $S$ is composed of bundled similar vectors, any future input evaluating $\text{Sim}(S_{\text{novel}}, S_{\tau-1}) > \epsilon$ natively triggers the transition pathway. Zero-shot generalization is mathematically guaranteed by the geometry.

### Level 1: Renormalizing Event Processor (Wernicke)

Level 1 continuously bundles the Level 0 sequence vector to track the macroscopic 'context' of a sentence.

$$
B_\tau = \text{sgn}\left( \sum_{t=0}^\tau \rho^{\tau-t}(S_t) \right)
$$

**Fast Structure Learning (Event Chunking):**

The model boundary-detects by tracking the differential similarity of the incoming token against the running bundle.

$$
\Delta_{\text{Sim}} = \text{Sim}(S_\tau, B_{\tau-1})
$$

If $\Delta_{\text{Sim}} < \kappa$ (a hard hyperparameter threshold), the system concludes a structural sentence boundary has been crossed, finalizes $B_{\tau-1}$ as a macroscopic `Event Vector`, and flushes the vector tracker.

### Level 2: Sophisticated Inference (Frontal Cortex)

Level 2 observes the macroscopic `Event Vectors` and performs **Deep Tree Search** to minimize Expected Free Energy ($G$).

Instead of continuous gradients, planning in VSA is purely algebraic simulation. Given a current belief $B_0$ and a set of candidate policy Actions $\{A_1, A_2 ... A_k\}$, the system calculates the Virtual Future $F$:

$$
F_{k} = \text{sgn}(B_0 + A_k)
$$

The Expected Free Energy $G(\pi)$ mapping to the Goal Vector $G^*$ simplifies completely to inverse cosine distance:

$$
G(A_k) \approx - \text{Sim}(F_k, G^*)
$$

By performing a recursive tree search simulating multiple sequential bundles, the agent evaluates the topological proximity of distant sequential actions exclusively through geometric overlap.

---

## 5. Bayesian Model Reduction (Combating the Superposition Catastrophe)

The major limitation of HDC mathematics is the capacity bound. Bundling $K$ vectors natively injects noise. If $K$ exceeds the theoretical limit (usually dependent on $d$), the resulting vector collapses into pure white noise, and $\text{Sim}$ evaluations fail. This is the **Superposition Catastrophe**.

To achieve infinite streaming capability (as demonstrated in `[train_modern_english.py]`), DHAI-4 integrates **Bayesian Model Reduction (BMR)**, implemented as a `Sleep/Consolidation Phase`.

The generative model periodically iterates over its stored Hebbian transition matrices $\mathbf{\Gamma}$. It applies an iterative hard-thresholding operation relative to an active epistemic pruning parameter $\psi$:

$$
\forall (\alpha \to \beta) \in \mathbf{\Gamma}: \text{if } \text{count}(\alpha \to \beta) < \psi, \text{prune}(\alpha \to \beta)
$$

This offline "sleep" operation structurally orthogonalizes the semantic graph, removing superimposed stochastic noise caused by single-occurrence edge-cases in the Wikipedia stream, thereby averting mathematical collapse and reclaiming infinite dimensionality.

---

## 6. Execution & Installation

The DHAI-4 repository operates entirely sequentially via zero-budget local compute, eschewing all backpropagation hardware. 

To observe the architecture constructing syntactic and semantic VSA matrices natively while streaming random modern Wikipedia articles:

```bash
# Clone the foundational architecture
git clone https://github.com/RatnaAnimesh/hai-txt.git
cd hai-txt

# Execute infinite Modern English HDC training
python dhai4_hdc/train_modern_english.py
```

*Observe the system parsing string bytes $\to$ generating $d=10,000$ matrices $\to$ calculating orthogonal bindings $\to$ executing Bayesian sleep cycles.*

---
**Author**: Animesh Ratna
**Context**: "Zero to Quant Hero" - Pillar III Research Thesis
