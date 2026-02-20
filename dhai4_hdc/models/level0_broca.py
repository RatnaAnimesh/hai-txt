import numpy as np
from dhai4_hdc.utils.hdc_math import get_hd_space

class HDC_SensoryInterface:
    def __init__(self, dimension=10000, role_sim_threshold=0.15):
        """
        Level 0: Sensory Interface (Broca).
        Implements Structural Factorization via VSA Binding (Role x Filler).
        """
        self.hd_space = get_hd_space(dimension)
        
        # Unsupervised Role Bootstrapping
        # We replace hardcoded Grammar (SUBJECT/VERB) with dynamic discovery.
        self.role_sim_threshold = role_sim_threshold
        self.word_contexts = {} # Maps word string -> its accumulated historical context bundle
        self.role_clusters = [] # List of discovered Role centroids
        self.prev_word_vec = None
        
        # Semantic Graph (B_semantic): sparse counts of bound concepts
        self.transition_counts = {}
        
        # Decoding Map: Maps bound vector bytes -> original raw word 
        # (Required for the model to "speak" conceptually)
        self.bound_to_word = {}

    def _vec_key(self, vec: np.ndarray) -> bytes:
        return vec.tobytes()

    def determine_role(self, word: str, filler_vec: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Unsupervised Role Assignment via Topological Sequence Clustering.
        A word's role is determined by grouping it with other words that share
        similar preceding contexts (where context is the permuted previous word).
        """
        # 1. Calculate sequential context via Permutation (\rho)
        if self.prev_word_vec is None:
            context_vec = self.hd_space.generate_atomic_vector("SEQUENCE_START")
        else:
            context_vec = self.hd_space.permute(self.prev_word_vec, shifts=1)

        # 2. Update the word's historical context bundle
        if word not in self.word_contexts:
            self.word_contexts[word] = context_vec
        else:
            self.word_contexts[word] = self.hd_space.bundle([self.word_contexts[word], context_vec])
            
        current_word_context = self.word_contexts[word]
        
        # 3. Find the closest Dynamic Role Cluster
        best_sim = -1.0
        best_role_idx = -1
        
        for idx, role_centroid in enumerate(self.role_clusters):
            sim = self.hd_space.similarity(current_word_context, role_centroid)
            if sim > best_sim:
                best_sim = sim
                best_role_idx = idx
                
        # 4. Bootstrap a new Role if context is entirely novel
        if best_sim < self.role_sim_threshold:
            new_role_vec = np.copy(current_word_context)
            self.role_clusters.append(new_role_vec)
            best_role_idx = len(self.role_clusters) - 1
            role_vec = new_role_vec
        else:
            # Update the existing role cluster centroid
            role_vec = self.role_clusters[best_role_idx]
            self.role_clusters[best_role_idx] = self.hd_space.bundle([role_vec, current_word_context])
            
        role_label = f"ROLE_{best_role_idx}"
        
        # Save filler vec for the NEXT word's context operation
        self.prev_word_vec = filler_vec
        
        return role_vec, role_label

    def encode(self, word: str) -> dict:
        """
        Encode a word by binding it to its dynamically discovered syntactic role.
        """
        # 1. Get raw Semantic Filler
        filler_vec = self.hd_space.generate_atomic_vector(word)
        
        # 2. Discover / Assign Syntactic Role Unsupervised
        role_vec, role_label = self.determine_role(word, filler_vec)
        
        # 3. BINDING (Structural Factorization)
        # The resulting vector is perfectly orthogonal to both Role and Filler
        bound_state = self.hd_space.bind(role_vec, filler_vec)
        
        # Store decoding map
        self.bound_to_word[self._vec_key(bound_state)] = word
        
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

    def sleep_cycle(self, prune_threshold: int = 1, force_prune_rate: float = 0.05) -> int:
        """
        Bayesian Model Reduction (Sleep / Consolidation).
        Prunes weak geometric connections (Hebbian counts below threshold),
        orthogonalizing the semantic graph and reclaiming memory capacity.
        Returns the number of weak nodes pruned.
        """
        pruned_graph = {}
        nodes_before = sum(len(transitions) for transitions in self.transition_counts.values())
        
        for p_key, transitions in self.transition_counts.items():
            strong_transitions = {}
            for k, v in transitions.items():
                if v > prune_threshold:
                    strong_transitions[k] = v
                elif v == prune_threshold:
                    # Probabilistic pruning for things that just met the threshold, 
                    # ensuring some N-grams survive to link longer geometries.
                    if np.random.random() > force_prune_rate:
                        strong_transitions[k] = v
                        
            if strong_transitions:
                pruned_graph[p_key] = strong_transitions
                
        self.transition_counts = pruned_graph
        nodes_after = sum(len(transitions) for transitions in self.transition_counts.values())
        
        return nodes_before - nodes_after

        return nodes_before - nodes_after

    def generate_sentence(self, max_length=20, temperature=1.5) -> str:
        """
        Generates a sentence by performing a weighted random walk over the learned
        geometric transition graph. Decodes the bound vectors back into raw words.
        Temperature encourages exploring slightly less frequent connections to build longer sentences.
        """
        if not self.transition_counts:
            return ""
            
        # Start random walk at a random known vector
        current_key = np.random.choice(list(self.transition_counts.keys()))
        sentence = [self.bound_to_word.get(current_key, "<UNK>")]
        
        for _ in range(max_length - 1):
            if current_key not in self.transition_counts or not self.transition_counts[current_key]:
                break # Reached a dead end geometrically
                
            transitions = self.transition_counts[current_key]
            next_keys = list(transitions.keys())
            counts = list(transitions.values())
            
            # Weighted probability based on learned Hebbian counts modified by temperature
            adjusted_counts = np.array(counts, dtype=np.float32) ** (1.0 / temperature)
            probabilities = adjusted_counts / np.sum(adjusted_counts)
            
            next_key = np.random.choice(next_keys, p=probabilities)
            word = self.bound_to_word.get(next_key, "<UNK>")
            sentence.append(word)
            current_key = next_key
            
            if word in {".", "!", "?"}: # Sentence bounded
                break
                
        # Clean up punctuation spacing natively
        text = " ".join(sentence)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" '", "'")
        return text
