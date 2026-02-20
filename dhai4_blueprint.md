# Toward Artificial General Intelligence: A Technical Blueprint for the Deep Hierarchical Active Inference (DHAI-4) Architecture via Structural Factorization and Renormalizing Generative Models

## 1. Introduction: The Dialectic of Structure and Scale in AGI

The trajectory of Artificial General Intelligence (AGI) research has historically oscillated between two distinct epistemological poles: the connectionist paradigm, which prioritizes scalable, high-dimensional parameter optimization (exemplified by the Transformer architecture), and the structured, neuro-symbolic paradigm, which emphasizes interpretable, hierarchical representations of the world. The recent retrospective analysis of the Deep Hierarchical Active Inference (DHAI-3) experiment marks a critical inflection point in this dialectic. The DHAI-3 project, an ambitious attempt to construct a "brain-replicating" language agent capable of organic development—from the rudimentary babbling of an infant to the sophisticated prose of a scholar—encountered a terminal barrier known as the Curse of Dimensionality. Despite successfully navigating early developmental stages, the agent succumbed to "Fluent Incoherence" in its mature phase, a failure mode characterized by rhythmic, poetic output devoid of logical semantic grounding.

This failure was not merely an engineering setback but a profound theoretical signal. It demonstrated that naive discrete Active Inference, while biologically inspired, cannot scale to the combinatorial magnitude of natural language without a fundamental architectural evolution. The brain does not simply "count" co-occurrences in a monolithic state space; it employs a highly factorized, hierarchical, and renormalizable strategy to manage the complexity of the environment.

This report serves as both a forensic autopsy of the DHAI-3 failure and a comprehensive architectural blueprint for its successor, theoretically designated DHAI-4. We posit that the path to AGI lies not in simply scaling parameters, but in Structural Factorization—a principle often referred to as the "Chomsky Fix"—which mathematically decouples Syntax (the rules of transition) from Semantics (the content of states). Furthermore, we integrate cutting-edge theoretical advancements in Renormalizing Generative Models (RGMs) and Hyperdimensional Computing (HDC). By synthesizing the interpretability of Active Inference with the deterministic, non-connectionist representational power of Vector Symbolic Architectures (VSA), we propose a unified methodology to overcome the sparsity constraints that doomed DHAI-3, generating semantic geometry natively without gradient descent.

## 2. Forensic Analysis of the DHAI-3 Architecture

To build a robust successor, we must first rigorously define the mathematical and structural failure modes of the DHAI-3 experiment. The project was grounded in the Free Energy Principle (FEP), utilizing a HierarchicalAgent with three cognitive strata: Level 0 (Sensory/Broca's Area), Level 1 (Semantic/Wernicke's Area), and Level 2 (Narrative/Prefrontal Cortex). While this biomimetic structure was theoretically sound, its implementation relied on discrete Hebbian learning over a monolithic state space, leading to inevitable collapse under the weight of high-dimensional natural language.

### 2.1 The Mathematical Failure Mode: Matrix Explosion and the Combinatorial Cliff

The terminal failure of DHAI-3 was a result of intrinsic mathematical intractability inherent in non-factorized discrete state spaces. In a discrete state-space model, transitions between states are governed by a transition matrix $\mathbf{B}$, where $B_{ij} = P(s_{t+1}=i | s_t=j)$. The learnability of this matrix is a function of the ratio between the parameter space size and the density of observed transitions in the training corpus.

#### 2.1.1 The Quadratic Expansion of Parameter Space
For Stage 1 ("The Baby"), the vocabulary $V$ was approximately 2,000 words. The resulting transition matrix size was manageable:
$$N_{params} = V \times V = 2,000 \times 2,000 = 4 \times 10^6 \text{ (4 million entries)}$$

At this scale, the "TinyStories" corpus provided sufficient density to populate the matrix. The agent successfully learned high-probability transitions (e.g., "The" $\rightarrow$ "cat"), mimicking the early syntax acquisition of a human child.

However, for Stage 2 ("The Student"), the vocabulary expanded to 30,000 words to accommodate the "FineWeb-Edu" curriculum. The matrix size exploded quadratically:
$$N_{params} = 30,000 \times 30,000 = 9 \times 10^8 \text{ (900 million entries)}$$

This creates a parameter space nearly 1,000 times larger. Crucially, while the parameter space expanded quadratically ($N^2$), the number of unique, meaningful bigrams in the training data did not scale proportionally. This led to a "Combinatorial Cliff" where the density of the probability matrix dropped below the threshold required for effective Hebbian consolidation.

#### 2.1.2 Zipf’s Law and the Sparsity Trap
The fundamental statistical property of natural language is its adherence to a Zipfian distribution, where the frequency of a word is inversely proportional to its rank ($f \propto 1/r$). A small set of function words ("the", "is", "of") appear millions of times, while the vast majority of content words ("aether", "neuronal", "epistemology") appear rarely.

In the DHAI-3 transition matrix, this distribution manifested as extreme, non-uniform sparsity:
*   **High-Frequency Zone**: The sub-matrix corresponding to function words was hyper-saturated. The agent over-learned these transitions, leading to strong local syntax.
*   **The Sparsity Desert**: The connections between rare content words (e.g., "Introduction" $\rightarrow$ "Methodology") were effectively zero because those specific pairs might never appear continuously in the training corpus.

Diagnostic analysis of DHAI-3 revealed a sparsity rate of **99.997%**. The model was essentially a lookup table with billions of empty slots. When the agent encountered a rare word during generation, the row in the transition matrix corresponding to that word was likely empty or undefined. Unlike a vector-based Neural Network, which can rely on the geometric proximity of "Introduction" and "Preface" to generalize, the discrete state encoding treated every word as an orthogonal, independent entity. Lacking a valid transition, the agent was forced to fall back on high-entropy random sampling or revert to the global prior (function words), destroying semantic coherence.

### 2.2 Phenomenology of Failure: Fluent Incoherence

"Fluent Incoherence" describes the specific behavioral artifact observed in Stage 3 ("The Scholar"). The agent, trained on 250 million tokens of Classic Literature, captured the rhythmic "vibe" and poetic vocabulary of the authors but failed to construct meaningful or logical sentences.

This phenomenon arises from the decoupling of hierarchical levels:
*   **Level 0 (Syntax/Broca)**: Operated at a fast timescale. Due to the high frequency of function words, this level successfully learned local grammatical rules (e.g., "The [Adjective][Noun]").
*   **Level 1 (Semantics/Wernicke)**: Operated at a slower timescale, intended to track "Concepts." However, due to matrix sparsity, the "downward" signals—priors sent from Level 1 to Level 0 to constrain word selection—were too diffuse to be effective.

**The Disconnect**: Without strong top-down semantic constraint, Level 0 operated in a "free-running" mode. It produced grammatically valid structures filled with statistically random content words, much like Wernicke's aphasia in humans.

The failure of DHAI-3 confirms that naive discrete Active Inference cannot scale to natural language without structural modification. The brain does not store a $30,000 \times 30,000$ matrix. It utilizes a factorized representation where Syntax (rules) and Semantics (content) are handled by distinct, interacting systems. This necessitates the "Chomsky Fix."

## 3. The Theoretical Solution: Structural Factorization (The "Chomsky Fix")

To remedy the combinatorial explosion, we must move from a Monolithic Generative Model to a Factorized Generative Model. This approach, colloquially termed the "Chomsky Fix," aligns with the linguistic theory that grammatical structure (Syntax) is independent of specific lexical items (Semantics). By mathematically decoupling these components, we can reduce the parameter space from $O(V^2)$ to $O(G^2 + M)$, where $G$ is the size of the grammar and $M$ is the size of the semantic map.

### 3.1 Mathematical Formulation of Structural Factorization

In a standard Hidden Markov Model (HMM) or POMDP used in Active Inference, the joint probability is defined as:
$$P(o, s) = P(s_1) \prod_{t=2}^T P(s_t | s_{t-1}) \prod_{t=1}^T P(o_t | s_t)$$
Where $P(s_t | s_{t-1})$ represents the massive $V \times V$ transition matrix $\mathbf{B}$.

Structural Factorization splits the hidden state $s_t$ into two distinct, interacting factors:
1.  **Syntactic State ($g_t$)**: Represents grammatical roles or high-frequency functional categories (e.g., Determinant, Noun-Phrase-Start, Verb-Transitive). The state space size $|G|$ is small and bounded (e.g., $\approx 1,000$ tags or function words).
2.  **Semantic State ($m_t$)**: Represents the "meaning," concept, or entity (e.g., CAT, TRUTH, RUN). The state space $|M|$ is large, but transitions are sparse and driven by narrative logic rather than adjacency.

The factored transition probability is approximated as:
$$P(s_t | s_{t-1}) \approx P(g_t | g_{t-1}) \times P(m_t | m_{t-1}, g_t)$$

#### 3.1.1 The Dense Syntax Matrix ($\mathbf{B}_{syntax}$)
The syntax transition matrix governs the flow of grammatical structure.
*   **Dimensions**: $|G| \times |G|$ (e.g., $1,000 \times 1,000 = 10^6$ parameters).
*   **Properties**: This matrix is dense and high-frequency. Because grammar rules (like "Determinant" $\rightarrow$ "Adjective") are universal across the corpus, this matrix is easily learnable even with limited data.
*   **Function**: It handles the "babbling" and rhythmic structure of language, ensuring that the output is always grammatically well-formed, regardless of semantic content.

#### 3.1.2 The Sparse Semantic Map ($\mathbf{B}_{semantic}$)
The semantic transition matrix is conditional on the syntax but evolves according to "concept space" dynamics.
*   **Decoupling Mechanism**: Crucially, the rule "Subject $\rightarrow$ Verb" is stored in $\mathbf{B}_{syntax}$. The specific realization ("Cat $\rightarrow$ Meows" vs. "Dog $\rightarrow$ Barks") is handled by the interaction between the semantic state and the syntactic slot.
*   **Knowledge-Agnostic Structure**: This aligns with the architecture of the DigiMind Synthesis Decoder (Dsynth). The Dsynth utilizes permanently frozen base weights for its syntactic and multimodal fusion tasks. This implies that the "rules of combination" (syntax) are fixed and optimized once, while the content (semantics) is dynamically retrieved from sparse specialist modules.
*   **Implication**: This factorization allows the agent to learn a new word (e.g., "Ocelot") by simply mapping it to a Semantic State (e.g., $m_{animal}$), instantly inheriting all the syntactic rules (runs, eats, sleeps) associated with that state, without needing to relearn the transitions for the new word itself.

### 3.2 Implementing the Factorization in Active Inference
In the Active Inference framework, this factorization is implemented via Mean-Field Approximation in the variational posterior. The agent maintains separable beliefs about syntax and semantics:
$$Q(s_t) \approx Q(g_t) \otimes Q(m_t)$$

This separation allows the agent to perform dual-track inference:
*   **Fast Track (Level 0)**: Updates $Q(g_t)$ based on immediate token observations, ensuring syntactic validity.
*   **Slow Track (Level 1)**: Updates $Q(m_t)$ based on the sequence of semantic content, maintaining narrative consistency.

By factorizing the generative model, we solve the Sparsity Trap. The Syntax Matrix is dense and robust. The Semantic Map is sparse but structured, accessed only when the Syntax Matrix dictates a "Content Slot" is open. This mirrors the neurobiological separation of Broca's area (syntax/sequencing) and Wernicke's area (lexical semantics).

## 4. Scaling Intelligence: Renormalizing Generative Models (RGMs)

While Structural Factorization solves the sparsity issue at a single level, it does not inherently solve the problem of long-range dependencies or "context." To achieve AGI, the agent must be able to process information across vast temporal scales—from phonemes to paragraphs to entire narratives. The Renormalizing Generative Model (RGM) provides the engine for this infinite scalability.

### 4.1 The Physics of Intelligence: Renormalization Group Theory
Friston et al. (2025) introduced RGMs as discrete homologs of Deep Neural Networks, grounded in the statistical physics of the Renormalization Group (RG). In physics, RG transformations allow for the description of a system at different scales by "coarse-graining" microscopic details into macroscopic effective degrees of freedom.

In the context of the DHAI-4 architecture, the RG operator aggregates fine-grained states (tokens) into coarse-grained events (concepts/gist):
*   **Micro-Scale ($L_0$)**: High-frequency fluctuations (Sensory data/Tokens).
*   **Macro-Scale ($L_1, L_2, \dots$)**: Slow-moving latent variables (Events/Narrative Arcs).

### 4.2 Scale-Invariance and Compositionality
The power of RGMs lies in their Scale Invariance. The model architecture is self-similar across hierarchical levels. The mathematical rules that govern word-to-word transitions ($L_0$) are identical to those governing sentence-to-sentence transitions ($L_1$), merely operating on a compressed temporal scale.

#### 4.2.1 Paths and Orbits as Latent Variables
Standard Markov models define transitions between static states ($s_t \rightarrow s_{t+1}$). RGMs generalize this by introducing Paths ($\zeta$) as latent variables.
*   **Definition**: A "Path" is a sequence or trajectory of transitions.
*   **Mechanism**: The model learns a probability distribution over paths, not just states.
*   **Deep Temporal Depth**: High-level states do not simply predict the next low-level state; they predict a trajectory or orbit of lower-level states. For example, a high-level state "Tragedy" entails a path of lower-level states "Loss $\rightarrow$ Grief $\rightarrow$ Resolution."

This mechanism directly addresses the "Fluent Incoherence" of DHAI-3. In the previous iteration, Level 1 predicted isolated "Concepts" but failed to enforce a trajectory. In an RGM, Level 1 generates a Path for Level 0 to follow, constraining the syntax to adhere to the narrative arc and ensuring long-horizon consistency.

### 4.3 Fast Structure Learning (Algorithm 1) and Event Discovery
One of the most critical contributions of the RGM framework is Fast Structure Learning. Unlike Deep Learning, which requires millions of iterations to "carve" structure from random weights, RGMs can identify structure through the coarse-graining of events.

*   **Compression Principle**: The model groups sequences of observations into discrete "events" based on maximizing mutual information (minimizing Variational Free Energy).
*   **Implementation Example**: In music analysis, RGMs successfully compressed a 2-minute jazz piece into 16 distinct "events" (bars). The model learned that specific bars follow others, effectively discovering the "syntax" of jazz improvisation without supervision.
*   **Application to Language**: In DHAI-4, this algorithm allows the agent to autonomously discover that sequences like "Subject-Verb-Object" or "If-Then" clauses form stable event units. This dynamically creates the high-level tokens for the "Semantic State" space ($M$), ensuring it is populated only by meaningful, statistically significant clusters. This is the Active Selection of model structure—a capability absent in static neural networks.

## 5. Hyperdimensional Computing: Native Semantic Geometry

We must gut the neural network components (LLM embeddings and rSLDS) to establish a system that natively generates its own semantic geometry without learning it through gradient descent. To do this, we pivot fully to Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA). This provides the exact same "generalization" superpower that neural networks have, but strictly through deterministic linear algebra and probability, keeping the architecture 100% non-connectionist.

### 5.1 The Geometry of HDC: Randomness as a Feature
In standard neural networks, embeddings are painstakingly learned so that "Cat" and "Dog" end up near each other in the vector space. HDC bypasses this entirely by exploiting the bizarre properties of high-dimensional geometry.

We define the state space not as integers (1, 2, 3), but as massive, hyperdimensional vectors (e.g., $d=10,000$). Every atomic base concept (a letter, or a root word) is assigned a completely random vector consisting of +1s and -1s.

Because the space is so vastly high-dimensional, any two randomly generated vectors are quasi-orthogonal (their dot product is extremely close to zero).
*   **Vector C**: `[+1, -1, -1, +1, ... ]` (Random representation of "Cat")
*   **Vector D**: `[-1, -1, +1, -1, ... ]` (Random representation of "Dog")

### 5.2 The Algebra of Thoughts (No Training Required)
Instead of a neural network learning how words combine, HDC uses strict mathematical operations to build complex concepts out of the random base vectors. This is the Algebra of Thoughts.

#### Binding (Multiplication/XOR)
Binding connects two concepts together, like putting a value into a variable. Mathematically, it is the element-wise multiplication of two vectors.
$$V_{bound} = Role_{Noun} \otimes Filler_{Cat}$$

**The Magic**: The resulting vector $V_{bound}$ is mathematically orthogonal (dissimilar) to both the Role and the Filler. It creates a brand new, unique state in the space. This is how the agent natively encodes structure (the "Chomsky Fix") without parameter explosion.

#### Bundling (Addition/Superposition)
Bundling combines multiple vectors into a set. It is simple vector addition, followed by a threshold to keep the values as +1 or -1.
$$V_{set} = Cat + Dog + Ocelot$$

**The Magic**: The resulting vector $V_{set}$ is highly similar (high dot product) to all of its constituent vectors. This automatically creates semantic clusters (e.g., the concept of "Feline") natively.

#### Permutation (Shift)
Permutation handles sequence and time. It is a simple cyclical shift of the vector coordinates, denoted by $\rho$.
$$V_{sequence} = \rho(The) + \rho(\rho(Cat))$$

This ensures that "The Cat" is mathematically distinct from "Cat The".

### 5.3 Integrating HDC with the Active Inference POMDP
By replacing one-hot integers with these HDC vectors, we completely fix the sparsity trap that caused the "Fluent Incoherence". Here is how the continuous vector math maps directly back to the discrete Hebbian counting of original Active Inference:

*   **The Observation ($o$)**: The incoming sensory data (a word) is projected into its pre-assigned, random HDC vector.
*   **The State Space ($s$)**: The hidden states in the POMDP are no longer just arbitrary numbers; they are HDC vectors representing bound concepts (e.g., $Noun \otimes Cat$).
*   **Generalization without Gradients**: If the agent learns a syntax rule for $V_{set}$ (the "Feline" bundle), and later encounters the word "Ocelot," it compares the Ocelot vector to its known states. Because "Ocelot" is mathematically similar to the bundle, the agent instantly applies the correct transition probability from the $\mathbf{B}$ matrix.

## 6. The Learning Mechanism: Hebbian Learning Remains Intact

Because HDC vectors can be compared using simple cosine similarity or Hamming distance, your original Dirichlet/Hebbian update rule remains perfectly valid. 

When the agent successfully predicts the next sequence, it updates the concentration parameters $\alpha$ in its transition matrices based on the similarity of the active HDC vectors. We are still just "counting coincidences" (updating Dirichlet priors), but we are counting them in a geometric space that naturally understands similarity. 

This entirely bypasses the need for a connectionist neural network, backpropagation (BPTT), or localized gradient-based learning. Training happens instantly via single-shot exposure by binding and bundling the observations into the transition geometry.

## 7. Safety and Planning: Language-Mediated Sophisticated Inference

The final pillar of the DHAI-4 architecture is the "Language-Mediated" layer. As proposed in recent frameworks by Wen (2025), using Natural Language as the state representation itself offers a path to safer, more interpretable, and aligned AGI.

### 7.1 HDC Associative Memory as the Belief Substrate
In traditional Active Inference, beliefs are represented as abstract probability distributions (Dirichlet or Gaussian). In this fully non-connectionist model, the belief state is explicitly represented as an HDC vector representing a natural language sentence.
*   **Observation Model (A)**: Represented as bound natural language hypotheses.
*   **Transition Model (B)**: Represented as causal narratives.
*   **Benefit**: Instead of leveraging an LLM, the A and B matrices are populated via HDC associative memory models. The agent uses pure symbolic lookup and associative geometric binding to retrieve causal transition logic: "What usually happens after dropping a glass?" The mathematically deterministic retrieval serves as the transition probability distribution for rigorous planning.

### 7.2 Safety via Belief-Preference Separation (The "Language Game" Fix)
A major risk in current LLM-based agents is Language Game Decoupling—the phenomenon where a system learns to talk safely (e.g., "I should not harm humans") but act unsafely because its internal reward functions are opaque.
*   **Solution**: The Language-Mediated framework enforces an explicit separation of Beliefs (what is true about the world) from Preferences (what is desired).
*   **Implementation**: In the Free Energy functional, these are mathematically distinct terms:
    *   **Ambiguity/Epistemic Value**: $D_{KL}[Q(s)||P(s)]$ (Belief update).
    *   **Risk/Pragmatic Value**: $\mathbb{E}_{Q}[\ln P(o|s) - \ln P(o)]$ (Preference satisfaction).
*   **Auditability**: By forcing the agent to articulate its belief ("I believe the user wants X") separate from its preference ("I want to minimize harm"), we create an auditable "Mind" that can be corrected by human oversight before action is taken.

### 7.3 Sophisticated Inference and Narrative Consistency
To solve the "Fluent Incoherence" of DHAI-3, the agent must employ Sophisticated Inference.
*   **Definition**: Sophisticated Inference is planning that considers "beliefs about future beliefs." It is a recursive tree search over the information state of the agent.
*   **Narrative Planning**: The agent asks: "If I introduce this character now, will it resolve the plot arc 100 steps later?"
*   **Expected Path Integral**: The agent minimizes the path integral of Expected Free Energy over a deep horizon. This ensures that current actions (syntax) are selected not just for their immediate probability, but for their contribution to the long-term resolution of the "Gist" (Level 2 state).

### 7.4 Validation: The ARC Benchmark
The safety and reasoning capabilities of this Language-Mediated architecture are validated using the Abstraction and Reasoning Corpus (ARC) benchmark. ARC tests the ability to learn new concepts from few examples (few-shot generalization) and is resistant to memorization. The framework's success on ARC demonstrates its ability to perform explicit reasoning and hypothesis testing, verifying that the "beliefs" it articulates are indeed the drivers of its behavior.

## 8. Conclusion: The Unified DHAI-4 Specification

Based on this exhaustive analysis, we propose the DHAI-4 architecture. This system abandons the naive discrete Hebbian approach of its predecessor in favor of a Hybrid, Factorized, Renormalizing, and Language-Mediated architecture.

### 8.1 Architectural Specification
*   **Level 0: The Factorized Sensory Interface (HDC Structure)**
    *   **Input**: High-dimensional bipolar/binary vectors ($d=10,000$) randomly assigned (atomic concepts).
    *   **Dynamics**: No gradients; abstract concepts are constructed via deterministic Binding ($\otimes$), Bundling ($+$), and Permutation ($\rho$).
    *   **Structural Factorization**:
        *   $\mathbf{B}_{syntax}$: A dense matrix operating over bound Role vectors.
        *   $\mathbf{B}_{semantic}$: A sparse graph mapping clustered semantic Bundles, updated via geometric Hebbian counting.

*   **Level 1: The Renormalizing Event Processor (RGM)**
    *   **Input**: Sequences of VSA semantic states from Level 0.
    *   **Function**: Uses Fast Structure Learning to bundle sequential tokens into "Events".
    *   **Latent Variable**: Paths ($\zeta$) modeled as complex VSA sequences predicting macroscopic trajectories.

*   **Level 2: Narrative Planner (HDC Sophisticated Inference)**
    *   **State**: HDC encoded Natural Language Beliefs.
    *   **Mechanism**: Language-Mediated Active Inference. It uses associative memory to retrieve potential narrative transitions.
    *   **Planning**: Performs a deep tree search (Sophisticated Inference) via HDC similarity comparisons to select the narrative structure that minimizes long-term surprise.

| Feature | DHAI-3 (Failed) | DHAI-4 (HDC Pivot Proposed) | Theoretical Advantage |
| :--- | :--- | :--- | :--- |
| **State Space** | Discrete (One-Hot) | Hyperdimensional Vectors (VSA) | Generalization without neural nets via geometric similarity. |
| **Transition Matrix** | Monolithic ($V \times V$) | Factorized ($G \times G$ + Bound Vectors) | Encoding logical structures without $O(N^2)$ scalar explosion. |
| **Learning Rule** | Naive Hebbian | Dirichlet/Hebbian over Cosine Sim | Single-shot learning natively tracks associative geometric closeness. |
| **Hierarchy** | Fixed Timescales (1, 4, 32) | Renormalizing (Scale-Free) | Adapts temporal depth to content; natively tracks VSA bundles. |
| **Planning** | Local Prediction | Sophisticated Inference (Deep Tree) | Enables long-horizon consistency (Narrative coherence). |
| **Safety** | Implicit | Explicit VSA Constraint | Fully deterministic algebraic representations, averting "black-box" LLM drifts. |

DHAI-4 represents a paradigm shift—moving from "learning distributions" to "learning structure." By implementing Structural Factorization, we allow the agent to master grammar without parameter explosion. By employing Renormalizing Generative Models, we give it the capacity to "grow" conceptual hierarchies that span from phonemes to plots. And by integrating Language-Mediated Active Inference, we ensure that this powerful system remains interpretable, safe, and aligned with human values. This is the theoretical threshold of AGI.

## 9. Hardening the Architecture: Limits of Infinite Scaling
As DHAI-4 transitions from prototype to infinite-scale training, four critical mathematical bounds of HDC must be addressed.

### 9.1 The Bootstrapping Problem: Unsupervised Role Assignment
Using heuristic or hardcoded grammatical roles ("SUBJECT", "VERB") leaks connectionist/statistical rules back into the architecture. DHAI-4 must learn these structurally. We implement a transition-based factorization where "Roles" are dynamically discovered bundles of words sharing the same temporal permutation ($\rho$). Fast Structure Learning will bind to these emergent geometric clusters autonomously.

### 9.2 Combating the Superposition Catastrophe
HDC capacity has a mathematically strict limit; adding (bundling) too many vectors eventually collapses the vector into orthogonal noise, ruining retrieval accuracy. To combat this, we implement **Bayesian Model Reduction (BMR) / Sleep Consolidation**. During offline downtime, the system iterates over its bundled transitions and applies a thresholding/thinning operation to prune weak connections, orthogonalizing concepts to reclaim capacity.

### 9.3 Tractable Deep Tree Search vs Combinatorial Bounds
Deep Tree Search in a 10,000-dimensional space causes rapid combinatorial explosion. We must temper this by using the **Epistemic Value** of Active Inference. The tree search dynamically prunes branches that do not actively reduce ambiguity regarding the Goal Vector.

### 9.4 Empirical Benchmarking
To objectively prove superiority over modern LLMs, the architecture will be validated on strict standard datasets:
- **BLiMP**: Benchmark of Linguistic Minimalist Pragmatics (to prove zero-shot grammatical geometry).
- **ARC**: Abstraction and Reasoning Corpus (to prove explicit top-down hypothesis generation vs statistical pattern matching).
