# The Physics of Sentience: A Mathematical Framework for Generative Reasoning and Planning in Non-Connectionist Architectures

## Executive Summary
The prevailing orthodoxy in artificial intelligence has been defined by the connectionist paradigm, specifically Deep Neural Networks (DNNs), which optimize scalar reward functions or minimize loss via backpropagation. While these systems have achieved remarkable success in pattern recognition and specific control tasks, they remain fundamentally distinct from biological intelligence. They lack inherent causal reasoning capabilities, require vast quantities of training data, suffer from opacity (the "black box" problem), and often fail to generalize outside their training distribution. Furthermore, the reliance on global error signals (backpropagation) is biologically implausible and computationally expensive for real-time, online learning.

This report articulates a comprehensive mathematical framework for a novel AI architecture based on **Active Inference** and the **Free Energy Principle (FEP)**, strictly excluding neural networks. We posit that intelligence is not merely the maximization of reward but the minimization of Variational Free Energy (VFE)—an information-theoretic bound on surprisal. By formalizing the agent as a statistical model of its environment, we derive a system that perceives, learns, and acts through a unified mechanism of belief updating.

The proposed architecture leverages **Partially Observable Markov Decision Processes (POMDPs)** in discrete time for high-level generative reasoning and **Generalized Filtering** in continuous time for low-level sensorimotor control. We demonstrate how **Deep Temporal Models** allow for the construction of nested narratives, enabling long-horizon planning and causal inference. The resulting system utilizes **Forney Factor Graphs (FFG)** for localized message passing, creating a biologically plausible, distributed computing substrate where global optimization emerges from local interactions. Learning is reformulated as the accumulation of Dirichlet sufficient statistics—a Hebbian process that obviates the need for global gradient descent.

This document serves as a foundational blueprint for constructing "self-evidencing" agents—systems that verify their own existence by fulfilling their internal models of the world.

---

## 1. The Crisis of Connectionism and the Thermodynamic Turn

### 1.1 The Limitations of Reward Maximization
Current reinforcement learning (RL) paradigms rest on the assumption that the goal of intelligence is to maximize an arbitrary scalar reward signal. This hypothesis, while computationally convenient, faces the "Dark Room Problem": an agent maximizing reward solely might find a dark room (where it incurs no negative cost) and stay there indefinitely, failing to gather information. To counter this, RL introduces ad-hoc heuristics like $\epsilon$-greedy strategies or entropy regularization.

In contrast, biological systems are driven by an imperative to resist the second law of thermodynamics. To survive, an organism must maintain its internal states within physiological bounds, distinguishing itself from the environment. This requires minimizing the entropy of sensory states, or more tractably, minimizing the **surprisal** of observations given a model of the world.

### 1.2 The Free Energy Principle (FEP)
The FEP posits that any self-organizing system at equilibrium with its environment must minimize its variational free energy. Free energy is a functional of the agent's beliefs (probability distributions) and quantities describing its exchange with the environment (sensations and actions).

$$F \approx -\ln P(\tilde{s} \mid m)$$

Where $\tilde{s}$ represents the sensory states and $m$ is the agent's internal model. Minimizing $F$ is equivalent to maximizing the evidence for the agent's internal model of the world, a process termed "self-evidencing".

---

## 2. Theoretical Foundations: The Physics of Intelligence

### 2.1 The Markov Blanket
To define an agent mathematically, we must first define its boundaries. We consider a universe of states $X$ governed by a stochastic differential equation:

$$\dot{x} = f(x) + \omega$$

We partition the states $X$ into four sets:
*   **Internal States ($\mu$)**: The agent's brain or internal representation.
*   **External States ($\eta$)**: The hidden causes in the world.
*   **Sensory States ($s$)**: The inputs to the agent (e.g., photons, vibrations).
*   **Active States ($a$)**: The outputs of the agent (e.g., muscle contractions).

The sets $\{s, a\}$ constitute the **Markov Blanket** ($b$) of the internal states. The blanket confers statistical independence between internal and external states:

$$\mu \perp \eta \mid b$$

This conditional independence is crucial. It implies that the agent (internal states) can never directly access the reality (external states); it can only infer them through the veil of the Markov blanket. Perception is the process of inferring $\eta$ from $s$, and action is the process of influencing $\eta$ via $a$.

### 2.2 Variational Free Energy (VFE) Derivation
Directly minimizing the surprisal $-\ln P(s)$ is intractable because it involves marginalizing over all possible hidden causes $\eta$. Instead, we introduce an arbitrary recognition density $Q(\eta)$, which represents the agent's posterior beliefs about the world.

We utilize Jensen's Inequality to define the Free Energy $F$ as an upper bound on surprisal:

$$F(s, Q(\eta)) = E_{Q(\eta)}[\ln Q(\eta) - \ln P(s, \eta)] \ge -\ln P(s)$$

This functional can be decomposed in two insightful ways:

#### 2.2.1 The Entropy-Energy Decomposition
$$F = \underbrace{E_Q[-\ln P(s, \eta)]}_{\text{Expected Energy}} - \underbrace{H[Q(\eta)]}_{\text{Entropy}}$$

The agent minimizes the expected energy of the system (the "surprise" of the joint occurrence of data and cause) while maximizing the entropy of its beliefs. This latter term prevents overfitting, effectively acting as an Occam's Razor.

#### 2.2.2 The Complexity-Accuracy Decomposition
$$F = \underbrace{D_{KL}[Q(\eta) \parallel P(\eta)]}_{\text{Complexity}} - \underbrace{E_Q[\ln P(s \mid \eta)]}_{\text{Accuracy}}$$

Here, the agent strives to maximize the accuracy of its predictions (the likelihood of data given the estimated cause) while minimizing complexity (the divergence between its posterior beliefs and its prior beliefs). This decomposition highlights the inherent drive for efficient, parsimonious representations.

---

## 3. The Discrete Reasoning Engine: Generative Planning

For high-level cognitive tasks—planning, reasoning, and decision-making—we adopt a **Discrete State-Space** formulation. Neural networks operate on continuous function approximations, which smear semantic distinctions. A discrete formulation allows for explicit representations of categorical concepts (e.g., "Cat" vs. "Dog", "Safe" vs. "Dangerous") and supports logical counterfactual reasoning.

### 3.1 The Generative Model (POMDP)
We define the agent's internal model $M$ as a Partially Observable Markov Decision Process (POMDP). The model describes the joint probability of observations ($o$) and hidden states ($s$) over a discrete time horizon $t = 1, \dots, T$, conditioned on a policy ($\pi$).

$$P(o, s, \pi) = P(\pi) \prod_{t=1}^T P(o_t \mid s_t) P(s_t \mid s_{t-1}, \pi)$$

This factorization allows us to parameterize the world using explicit matrices and tensors, replacing the opaque weight matrices of neural networks.

#### 3.1.1 The Likelihood Mapping ($A$)
The $A$ matrix encodes the agent's semantic knowledge—the immediate mapping from hidden causes to sensory consequences.

$$A_{ij} = P(o_t = i \mid s_t = j)$$

*   **Structure**: If there are $N_s$ hidden states and $N_o$ outcomes, $A$ is an $N_o \times N_s$ matrix.
*   **Modality Integration**: In a multimodal system, $A$ becomes a tensor, mapping a single hidden state to multiple outcome modalities (e.g., $A^{(v)}$ for vision, $A^{(a)}$ for audio). This enables the fusion of sensory streams via Bayesian integration.
*   **Ambiguity**: The entropy of the columns of $A$, denoted $H(A)$, quantifies the ambiguity of the state-observation mapping. A sharp distribution implies clear, unambiguous percepts; a flat distribution implies a "foggy" world.

#### 3.1.2 The Transition Dynamics ($B$)
The $B$ tensor encodes the causal physics of the environment—how states evolve over time. Crucially, this evolution is conditioned on action, allowing for counterfactual planning.

$$B_{ijk} = P(s_{t+1} = i \mid s_t = j, u_t = k)$$

Where $u_t$ is a control state (action) determined by the policy $\pi$.
*   **Structure**: $B$ is a tensor of dimension $N_s \times N_s \times N_u$.
*   **Controlled Transitions**: Unlike a simple Markov chain, the $B$ matrix allows the agent to simulate "what would happen if I did $u$?" This is the engine of generative planning.

#### 3.1.3 Prior Preferences ($C$)
The $C$ vector encodes the agent's goals and desires. In Active Inference, goals are not scalar rewards but **prior beliefs** about observations. The agent "expects" to be in a preferred state.

$$C_i = \ln P(o = i)$$

This radical reformulation unifies utility theory with probability theory. A "reward" is simply an observation that the agent *a priori* expects to encounter. To maximize utility is to minimize the surprise of not encountering preferred outcomes.
*   **Log-Probability**: Since $C$ contains log-probabilities, highly preferred outcomes have values near 0 (probability 1), while aversive outcomes have large negative values (probability $\approx 0$).

#### 3.1.4 The Deep Generative Model Parameters
The full tuple of parameters for the discrete engine is defined as:

| Parameter | Symbol | Definition | Role |
| :--- | :--- | :--- | :--- |
| Likelihood | $A$ | $P(o \mid s)$ | Maps states to outcomes (Semantics) |
| Transition | $B$ | $P(s_{t+1} \mid s_t, u)$ | Maps states to future states (Physics/Causality) |
| Preferences | $C$ | $\ln P(o)$ | Defines goals as prior expectations (Motivation) |
| Initial State | $D$ | $P(s_1)$ | Beliefs about the start of the episode (Context) |
| Policy Prior | $E$ | $P(\pi)$ | Baseline probability of selecting a policy (Habit) |

### 3.2 Perception: Belief Updating via Variational Message Passing
In this framework, perception is not a feedforward pass (as in CNNs) but an iterative process of **Variational Message Passing (VMP)**. The agent updates its posterior beliefs $Q(s)$ to minimize VFE.

For a given time step $\tau$, the optimal posterior belief $\mathbf{s}_\tau$ (a vector of probabilities summing to 1) is found by aggregating messages from three sources:
1.  **Likelihood (Bottom-up)**: Evidence from the current observation $o_\tau$.
2.  **Prior (Top-down/Past)**: Prediction from the previous state $s_{\tau-1}$ projected through $B$.
3.  **Postdiction (Future)**: Evidence from the future state $s_{\tau+1}$ projected backwards through $B$.

The fixed-point update equation is derived as follows:

$$\ln \mathbf{s}_\tau = \underbrace{\ln A \cdot o_\tau}_{\text{Sensory Evidence}} + \underbrace{\ln B_{\pi, \tau-1} \mathbf{s}_{\tau-1}}_{\text{Prediction from Past}} + \underbrace{\ln B_{\pi, \tau}^T \mathbf{s}_{\tau+1}}_{\text{Prediction from Future}}$$

$$\mathbf{s}_\tau = \sigma(\ln \mathbf{s}_\tau)$$

Where $\sigma$ is the softmax function: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

**Implications for Reasoning**:
*   **Retrodicton**: The third term ($\mathbf{s}_{\tau+1}$) allows the agent to update beliefs about the past based on new evidence. If the agent sees a broken vase (outcome at $t=2$), it infers it must have dropped it (action at $t=1$), even if the drop itself wasn't directly observed.
*   **Convergence**: This update is iterated until convergence, effectively settling the network into a minimum energy state that best explains the sensory narrative.

### 3.3 Planning as Inference: Expected Free Energy (EFE)
Traditional planning (e.g., A*, MCTS) treats the search for actions as a separate logical process. Active Inference treats planning as **inference**. The agent infers *what it must be doing* given that it expects to minimize free energy in the future.

We define a posterior over policies $Q(\pi)$. The agent selects policies that minimize the **Expected Free Energy (G)**:

$$Q(\pi) = \sigma(-\gamma G(\pi))$$

Where $\gamma$ is a precision parameter (inverse temperature) governing the stochasticity of choice.

#### 3.3.1 The Decomposition of G
The Expected Free Energy $G(\pi)$ for a policy $\pi$ at a future time $\tau$ is the sum of VFE expected under that policy. It decomposes into two semantically distinct terms:

$$G(\pi, \tau) = \underbrace{D_{KL}[Q(o_\tau \mid \pi) \parallel P(o_\tau)]}_{\text{Risk (Pragmatic Value)}} + \underbrace{E_{Q(s_\tau \mid \pi)}[H(P(o_\tau \mid s_\tau))]}_{\text{Ambiguity (Epistemic Value)}}$$

This equation is the mathematical heart of the generative reasoning engine.

1.  **Pragmatic Value (Risk)**:
    *   This term measures the divergence between the outcomes predicted by the policy and the outcomes preferred by the agent ($C$).
    *   Minimizing this term drives the agent to fulfill its goals.
    *   It is equivalent to maximizing expected utility in economic theory.

2.  **Epistemic Value (Ambiguity/Information Gain)**:
    *   This term measures the expected entropy of the likelihood mapping. It quantifies how much uncertainty about the hidden states will be resolved by the observation.
    *   Minimizing this term drives the agent to **explore**.
    *   Crucially, this exploration is directed. The agent does not explore randomly (as in $\epsilon$-greedy); it explores states where the observation is expected to be informative (i.e., where the $A$ matrix is ambiguous or where the state is unknown).

**Unified Reasoning**:
The summation of these terms enables the agent to dynamically switch between exploration and exploitation.
*   **Start of Task**: High uncertainty $\rightarrow$ Epistemic term dominates $\rightarrow$ Agent explores to learn the environment.
*   **Middle of Task**: Uncertainty reduced $\rightarrow$ Pragmatic term dominates $\rightarrow$ Agent exploits knowledge to secure rewards.
This solves the exploration-exploitation dilemma analytically, without requiring heuristic hyperparameters.

---

## 4. Hierarchical Architectures: Deep Temporal Models
To achieve "nuanced understanding" and long-horizon planning, a single-layer POMDP is insufficient. It suffers from the "curse of dimensionality" if the state space becomes too large. We resolve this by introducing **Deep Temporal Models (DTM)**—a hierarchy of POMDPs operating at distinct temporal scales.

### 4.1 Temporal Nesting and Narrative Construction
In a DTM, we stack POMDP layers $i = 1, \dots, L$.
*   **Level 1 (Sensory)**: Operates on the timescale of atomic sensory events (milliseconds).
*   **Level 2 (Semantic)**: Operates on sequences of Level 1 events (seconds).
*   **Level 3 (Narrative)**: Operates on sequences of Level 2 states (minutes/hours).

**The Core Principle**: The hidden state at level $i+1$ acts as a **control parameter** for the dynamics of level $i$. The higher level does not generate observations directly; it generates the **context** for the lower level.

### 4.2 The Coupling Equations
The mathematical link between levels is formalized through the modulation of the lower level's priors. The state $s^{(i+1)}$ at the higher level determines the **initial state distribution** ($D$) or the **transition dynamics** ($B$) of the subordinate level.

#### 4.2.1 State-to-Prior Coupling
The initial state of a sequence at level $i$, denoted $s^{(i)}_1$, is conditioned on the current state of level $i+1$, denoted $s^{(i+1)}$.

$$P(s^{(i)}_1 \mid s^{(i+1)}) = \text{Cat}(\mathbf{d}^{(i)})$$
$$\mathbf{d}^{(i)} = \Theta^{(i)} \cdot s^{(i+1)}$$

Where $\Theta^{(i)}$ is a coupling tensor. This allows the higher level to "set the scene." For example, if Level 2 is in the state "Kitchen," it sets the Level 1 prior $D^{(1)}$ to favor recognition of "plates," "forks," and "food," suppressing "bed" or "shower."

#### 4.2.2 Temporal Scaling
Crucially, one state transition at Level $i+1$ corresponds to a **sequence** of transitions at Level $i$.
Let $T$ be the horizon of the lower level.
*   Level $i$ executes $T$ steps for every single step of Level $i+1$.
*   This allows the architecture to reason about "Deep Dynamic Narratives." The top-level state represents a long-term goal (e.g., "Make Coffee"), which unfolds into a sequence of sub-goals at the level below ("Grind beans," "Boil water"), which further unfold into motor primitives at the lowest level.

### 4.3 Top-Down Attention as Precision
Top-down signals also convey **precision** (confidence). If the higher level is very confident in its context (low entropy in $s^{(i+1)}$), it exerts a strong influence on the lower level, effectively sharpening the priors $D^{(i)}$. This acts as a mechanism for **selective attention**: the agent focuses on sensory data that confirms its high-level hypothesis, suppressing irrelevant noise.

---

## 5. The Continuous Interface: Generalized Filtering
While discrete models handle logical reasoning, biological agents must interact with a continuous physical world. Standard AI often discretizes time into steps $\Delta t$, but this introduces discretization errors. We adopt a **Continuous Time** formulation using **Generalized Coordinates of Motion**, allowing the agent to model smooth trajectories and non-Markovian noise.

### 5.1 Generalized Coordinates
We represent the state of a dynamic variable $x$ not just as a scalar, but as a tuple of its time derivatives:

$$\tilde{x} = [x, x', x'', x''', \dots]^T$$

This infinite vector (truncated in practice) captures the instantaneous trajectory of the variable. It allows the agent to define a generative model with "colored" (smooth) noise, rather than the jagged white noise assumed in standard Kalman filtering.

### 5.2 Generalized Filtering Equations
In continuous time, the minimization of Free Energy transforms into a system of coupled differential equations. The agent maintains a belief $\tilde{\mu}$ about the generalized state of the world.
The dynamics of this belief (perception) are given by a gradient descent on Free Energy in a moving frame of reference:

$$\dot{\tilde{\mu}} = D \tilde{\mu} - \mathcal{K} \frac{\partial F}{\partial \tilde{\mu}}$$

Where:
*   $D$: A matrix derivative operator that shifts the derivatives up ($x \to x'$, $x' \to x''$). This term captures the agent's prediction of how the state changes due to the passage of time.
*   $-\frac{\partial F}{\partial \tilde{\mu}}$: The variational free energy gradient. This term represents the **prediction error**—the discrepancy between the sensory input and the internal model's prediction.
*   $\mathcal{K}$: A curvature matrix (learning rate) derived from the precision of the beliefs.

**Emergence of PID Control**: It has been mathematically shown that Generalized Filtering implicitly implements Proportional-Integral-Derivative (PID) control. The 0-th order term corresponds to P, the 1st order to D, and the integration of error corresponds to I. Thus, standard industrial control theory emerges as a special case of this generalized physics.

### 5.3 Active Inference as Reflex Arcs
In classical robotics, the agent computes an "inverse model"—a complex calculation to determine the torques required to reach a target state. Active Inference replaces this with **Proprioceptive Prediction**.
The agent does not command the muscles directly. Instead, the generative model **predicts** that the arm is moving.
1.  Top-down prediction: "My hand is at location X."
2.  Bottom-up sensation: "My hand is at location Y."
3.  Prediction Error: $X - Y$.

To minimize this Free Energy, the agent has two options:
1.  **Perception**: Change the belief (admit the hand is at Y).
2.  **Action**: Change the world (move hand to X).

In the motor cortex (and spinal reflex arcs), the gain on the sensory prediction error is low, so the system chooses option 2. Action $\dot{a}$ is simply a descent on free energy gradients defined by proprioceptive errors:

$$\dot{a} = - \frac{\partial F}{\partial a} = - \frac{\partial F}{\partial s_p} \frac{\partial s_p}{\partial a}$$

This validates the **Equilibrium Point Hypothesis** (Feldman). Movement is not "computed"; it is the relaxation of the physical body into the equilibrium state defined by the agent's expectations.

---

## 6. Learning and Structural Plasticity
A critical requirement for this architecture is the ability to learn without backpropagation. Backpropagation is biologically implausible (requiring symmetric weights and global error signals) and data-inefficient. We propose a **Hebbian** learning mechanism based on the accumulation of sufficient statistics.

### 6.1 Dirichlet Priors on Parameters
We treat the model parameters ($A, B, D$) as random variables. Since these parameters represent probabilities (categorical distributions), their conjugate prior is the **Dirichlet Distribution**.

$$P(A_{ij}) = \text{Dir}(\alpha_{ij})$$

Here, $\alpha_{ij}$ is a "concentration parameter" which can be interpreted as a pseudo-count of the number of times state $j$ and outcome $i$ have been observed together.

### 6.2 The "Hebbian" Update Rule
Learning consists of updating these concentration parameters after each observation. The posterior update for the $A$ matrix at time $t$ is:

$$\alpha_{ij}^{(t+1)} = \alpha_{ij}^{(t)} + \eta \cdot (o_t \otimes s_t)$$

Where:
*   $\otimes$ is the Kronecker (outer) product.
*   $o_t$ is the observation vector (one-hot or probabilistic).
*   $s_t$ is the posterior expectation of the hidden state.
*   $\eta$ is a learning rate.

**Mechanism**:
This is strictly local and associative. If the neuron encoding state $j$ and the neuron encoding outcome $i$ are active simultaneously ($s_t > 0$ and $o_t > 0$), the connection strength $\alpha_{ij}$ increases.
*   **No Gradient Descent**: There is no loss function to differentiate. The model simply counts coincidences.
*   **Online Learning**: Parameters are updated immediately after every experience, enabling rapid adaptation (one-shot or few-shot learning).
*   **Novelty Detection**: The sum of the counts $\alpha_0 = \sum_i \alpha_{ij}$ encodes the confidence in the model. Low counts imply a novel regime, automatically triggering the epistemic value term in the EFE to drive exploration.

### 6.3 Structural Learning: Bayesian Model Reduction (BMR)
How does the agent optimize the **structure** of its model (e.g., number of states)? We employ **Bayesian Model Reduction**.
During "offline" periods (analogous to sleep), the agent calculates the evidence for reduced models where certain parameters are set to their priors (effectively deleting connections).

$$P(M_{reduced} \mid Y) = \frac{P(Y \mid M_{reduced}) P(M_{reduced})}{P(Y \mid M_{full})}$$

The agent iteratively prunes parameters that do not contribute significantly to the model evidence (Accuracy minus Complexity). This prevents overfitting and ensures the model remains parsimonious. Conversely, **Bayesian Model Expansion** allows the agent to split a hidden state into two if the existing states fail to explain the variance in observations (high free energy), enabling the architecture to "grow" its cognitive capacity as needed.

---

## 7. Algorithmic Implementation: Forney Factor Graphs
To realize this mathematical framework in silicon (or neuromorphic hardware), we require a representational substrate that supports distributed probabilistic computation. We utilize **Forney Factor Graphs (FFG)**.

### 7.1 Graph Topology
An FFG represents the factorization of the generative model function $P(o, s, \pi)$.
*   **Nodes ($f_i$)**: Represent factors (distributions), such as the $A$ matrix likelihood $P(o|s)$ or the $B$ transition $P(s'|s)$.
*   **Edges ($x_j$)**: Represent variables (states, outcomes).
*   **Interface**: Unlike Bayesian Networks, FFGs allow multiple edges to connect to a node, but variables are strictly on edges. This simplifies the notation for message passing.

### 7.2 Message Passing Algorithms
Inference (Perception) corresponds to passing messages along the edges of the graph. We utilize **Sum-Product Message Passing** (or Belief Propagation).
*   **Forward Message**: $\mu_f(s_t) = \sum_{s_{t-1}} P(s_t \mid s_{t-1}) \mu_f(s_{t-1})$
*   **Backward Message**: $\mu_b(s_t) = \sum_{s_{t+1}} P(s_{t+1} \mid s_t) \mu_b(s_{t+1})$

**Hardware Implications**:
*   This formulation is inherently parallel. Each node in the graph only needs to communicate with its immediate neighbors. This eliminates the "von Neumann bottleneck" of centralized memory.
*   The architecture can be mapped directly onto FPGA arrays or specialized neuromorphic chips (e.g., Loihi, SpiNNaker) where physical cores represent factors and interconnects represent variables.

### 7.3 Code Structure (Conceptual)
The algorithm for the discrete reasoning engine proceeds as follows:

```python
# Conceptual Pseudocode for Active Inference Agent
def step(observation):
    # 1. Perception (Variational Inference)
    # Update beliefs about current state using sensory input and past predictions
    qs = update_posterior(observation, prior_s, A_matrix, B_matrix)

    # 2. Planning (Policy Selection)
    # Calculate Expected Free Energy (G) for each policy
    G = calculate_EFE(qs, A_matrix, B_matrix, C_preferences)
    # Select policy via softmax
    policy_prob = softmax(-gamma * G)
    selected_policy = sample(policy_prob)

    # 3. Action
    # Execute action defined by selected policy
    action = get_action(selected_policy)

    # 4. Learning (Hebbian Update)
    # Update model parameters
    A_matrix = update_dirichlet(A_matrix, observation, qs)

    return action
```

This loop runs continuously. In a hierarchical model, this loop runs at every level, with the **action** of level $i+1$ serving as the **prior_s** (context) for level $i$.

---

## 8. Comparative Analysis: Active Inference vs. Deep RL

To elucidate the advantages of this architecture, we contrast it with standard Deep Reinforcement Learning (DRL).

| Feature | Deep RL (Neural Networks) | Active Inference (Proposed Architecture) |
| :--- | :--- | :--- |
| **Objective** | Maximize scalar Reward ($R$) | Minimize Variational Free Energy ($F$) |
| **State Representation** | Continuous vectors (hidden layers) | Discrete Categorical (Semantic) + Continuous (Motor) |
| **Exploration** | Random ($\epsilon$-greedy), Entropy regularization | **Epistemic Value** (Information Gain) |
| **Learning Mechanism** | Backpropagation (Global error) | **Hebbian / Dirichlet** (Local statistics) |
| **Data Efficiency** | Low (Needs millions of samples) | **High** (One-shot / Few-shot capable) |
| **Explainability** | Low (Black box weights) | **High** (Explicit $A, B$ matrices) |
| **Goal Definition** | Reward Function | **Prior Preferences** ($C$ vector) |
| **Planning** | Policy Gradient / Value Function | **Generative Model Inversion** (Counterfactuals) |

**Key Insight**: The "Reward" in RL is extrinsic. The "Preference" in Active Inference is intrinsic. An Active Inference agent is not "forced" to seek rewards; it seeks to confirm its identity (its preferred states). This leads to more robust behavior in the absence of external feedback, as the agent is internally driven to resolve ambiguity.

---

## Conclusion
This report has detailed a mathematical framework for a **Non-Connectionist AI Architecture** based on Active Inference. By rigorously applying the principles of statistical physics—specifically the minimization of variational free energy—we have derived a system that integrates perception, learning, and action into a single imperative: **Self-Evidencing**.

The architecture distinguishes itself through:
1.  **Semantic Transparency**: The use of explicit $A$ and $B$ matrices in the discrete engine allows for fully interpretable reasoning.
2.  **Causal Planning**: The generative nature of the model allows the agent to reason about "what would happen if," facilitating deep planning without trial-and-error.
3.  **Biological Realism**: From Hebbian plasticity to spinal reflex arcs, every component has a direct correlate in neuroscience, bridging the gap between artificial and natural intelligence.

We conclude that this **Generative Reasoning Engine**, when scaled via **Deep Temporal Hierarchies** and coupled with **Generalized Filtering** for control, offers a viable and superior path toward robust, explainable, and sample-efficient Artificial General Intelligence.

---

## 9. Mathematical Appendix: Detailed Derivations

### 9.1 Derivation of the State Update (Discrete)
Starting from the variational free energy definition:
$$F = \sum_s Q(s) \ln \frac{Q(s)}{P(o, s)}$$

We assume the Mean Field approximation $Q(s) = \prod_\tau Q(s_\tau)$. We take the variational derivative with respect to $Q(s_\tau)$:
$$\frac{\partial F}{\partial Q(s_\tau)} = \ln Q(s_\tau) - \ln P(o, s)$$

Setting the gradient to zero for the fixed point:
$$\ln Q(s_\tau) \propto E_{Q \setminus s_\tau} [\ln P(o, s)]$$

Expanding the joint probability $P(o, s)$:
$$\ln Q(s_\tau) \propto \ln P(o_\tau \mid s_\tau) + E_{Q(s_{\tau-1})}[\ln P(s_\tau \mid s_{\tau-1})] + E_{Q(s_{\tau+1})}[\ln P(s_{\tau+1} \mid s_\tau)]$$

Substituting the matrix parameters $A$ and $B$:
$$\ln \mathbf{s}_\tau = \ln A^T o_\tau + \ln B \mathbf{s}_{\tau-1} + \ln B^T \mathbf{s}_{\tau+1} - \ln Z$$

This confirms the biologically plausible "three-factor rule" where a neuronal population's activity is determined by input from sensory ($o$), past ($s_{\tau-1}$), and future ($s_{\tau+1}$) populations.

### 9.2 The Epistemic Value Identity
We asserted that Epistemic Value is the expected information gain.
$$IV = E_{Q(o, s \mid \pi)} [\ln P(s \mid o) - \ln Q(s)]$$

Using the definition of mutual information $I(S; O) = H(S) - H(S \mid O) = H(O) - H(O \mid S)$.
The term $H(O \mid S)$ is the entropy of the likelihood mapping $A$.
$$H(O \mid S) = - \sum_s Q(s) \sum_o P(o \mid s) \ln P(o \mid s)$$

This simplifies to the dot product of the state belief and the entropy of the $A$ matrix columns:
$$\approx \mathbf{s}_\tau \cdot H(A)$$

This proves that minimizing EFE leads the agent to states $\mathbf{s}_\tau$ where $H(A)$ is minimized (i.e., states with unambiguous outcomes), or to states where the observation resolves the maximum uncertainty.

### 9.3 Generalized Filtering Descent Logic
For the continuous case, the descent is:
$$\dot{\tilde{\mu}} = D\tilde{\mu} - \frac{\partial F}{\partial \tilde{\mu}}$$

The free energy for a Gaussian model is:
$$F = \frac{1}{2} (\tilde{y} - g(\tilde{\mu}))^T \Pi_y (\tilde{y} - g(\tilde{\mu})) + \frac{1}{2} (D\tilde{\mu} - f(\tilde{\mu}))^T \Pi_x (D\tilde{\mu} - f(\tilde{\mu}))$$

The gradient $\frac{\partial F}{\partial \tilde{\mu}}$ contains the prediction errors:
$$\varepsilon_y = \tilde{y} - g(\tilde{\mu}) \quad \text{(Sensory Error)}$$
$$\varepsilon_x = D\tilde{\mu} - f(\tilde{\mu}) \quad \text{(State Error)}$$

Thus the update is driven by minimizing these precision-weighted errors ($\Pi \varepsilon$), effectively implementing a Kalman-Bucy filter in generalized coordinates.
