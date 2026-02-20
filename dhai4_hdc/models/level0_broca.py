import numpy as np
from dhai4_hdc.utils.hdc_math import get_hd_space

class HDC_SensoryInterface:
    def __init__(self, dimension=10000):
        """
        Level 0: Sensory Interface (Broca).
        Implements Structural Factorization via VSA Binding (Role x Filler).
        """
        self.hd_space = get_hd_space(dimension)
        
        # 1. Generate core abstract Role vectors (The Grammar)
        # These are fixed, orthogonal structural anchors
        self.roles = {
            "SUBJECT": self.hd_space.generate_atomic_vector("ROLE_SUBJECT"),
            "VERB":    self.hd_space.generate_atomic_vector("ROLE_VERB"),
            "OBJECT":  self.hd_space.generate_atomic_vector("ROLE_OBJECT"),
            "PUNCT":   self.hd_space.generate_atomic_vector("ROLE_PUNCT")
        }
        
        # 2. Syntax Matrix (B_syntax): dense transitions between Roles
        # In a real model, this is learned. Here we mock a basic English S-V-O loop.
        self.syntax_transitions = {
            "SUBJECT": "VERB",
            "VERB": "OBJECT",
            "OBJECT": "PUNCT",
            "PUNCT": "SUBJECT"
        }
        
        # 3. Semantic Graph (B_semantic): sparse counts of bound concepts
        self.transition_counts = {}
        
        # Naive state tracker for the demo (assumes S-V-O rigid structure)
        self.current_role = "SUBJECT"

    def _vec_key(self, vec: np.ndarray) -> bytes:
        return vec.tobytes()

    def determine_role(self, word: str) -> str:
        """
        In a purely unsupervised RGM, roles are discovered via clustering.
        For this prototype, we use a crude heuristic to cycle roles to demonstrate Binding.
        """
        if word in ['.', '!', '?']:
            self.current_role = "PUNCT"
            return "PUNCT"
            
        role = self.current_role
        # Advance the rigid syntax state machine
        self.current_role = self.syntax_transitions[role]
        if self.current_role == "PUNCT": # Skip punct if not actual punct
            self.current_role = "SUBJECT"
            
        return role

    def encode(self, word: str) -> dict:
        """
        Encode a word by binding it to its current syntactic role.
        Returns the bound state, the raw filler, and the role used.
        """
        # 1. Get raw Semantic Filler (the word itself)
        filler_vec = self.hd_space.generate_atomic_vector(word)
        
        # 2. Determine and get Syntactic Role
        role_label = self.determine_role(word)
        role_vec = self.roles[role_label]
        
        # 3. BINDING (Structural Factorization)
        # The resulting vector is mathematically independent of both the Role and Word
        bound_state = self.hd_space.bind(role_vec, filler_vec)
        
        return {
            "bound_state": bound_state,
            "filler": filler_vec,
            "role": role_label
        }

    def learn_transition(self, prev_bound_vec: np.ndarray, curr_bound_vec: np.ndarray):
        """
        Hebbian counting over the strictly bound semantic-syntactic geometry.
        """
        p_key = self._vec_key(prev_bound_vec)
        c_key = self._vec_key(curr_bound_vec)
        
        if p_key not in self.transition_counts:
            self.transition_counts[p_key] = {}
            
        if c_key not in self.transition_counts[p_key]:
            self.transition_counts[p_key][c_key] = 0
            
        self.transition_counts[p_key][c_key] += 1

    def sleep_cycle(self, prune_threshold: int = 2) -> int:
        """
        Bayesian Model Reduction (Sleep / Consolidation).
        Prunes weak geometric connections (Hebbian counts below threshold),
        orthogonalizing the semantic graph and reclaiming memory capacity.
        Returns the number of weak nodes pruned.
        """
        pruned_graph = {}
        nodes_before = sum(len(transitions) for transitions in self.transition_counts.values())
        
        for p_key, transitions in self.transition_counts.items():
            strong_transitions = {k: v for k, v in transitions.items() if v >= prune_threshold}
            if strong_transitions:
                pruned_graph[p_key] = strong_transitions
                
        self.transition_counts = pruned_graph
        nodes_after = sum(len(transitions) for transitions in self.transition_counts.values())
        
        return nodes_before - nodes_after

