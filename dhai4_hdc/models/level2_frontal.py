import numpy as np
from dhai4_hdc.utils.hdc_math import get_hd_space

class HDC_NarrativePlanner:
    def __init__(self, dimension=10000, search_depth=3, belief_window=5):
        """
        Level 2: Narrative Planner (Frontal Cortex).
        Uses HDC Associative Memory for Sophisticated Inference (Deep Tree Search).
        """
        self.hd_space = get_hd_space(dimension)
        self.goal_vector = None
        self.current_belief = None
        self.search_depth = search_depth
        self.belief_window = belief_window
        self.recent_events = []
        
        # A simple associative memory transition graph: Action -> [Possible Consequence Actions]
        self.action_memory = {}
        
    def set_goal(self, concept_words: list[str]):
        """
        Defines the narrative goal as a bundled vector of key concepts.
        """
        vecs = [self.hd_space.generate_atomic_vector(w) for w in concept_words]
        self.goal_vector = self.hd_space.bundle(vecs)
        
    def learn_action_transition(self, action1: str, action2: str):
        """
        Simulates learning that action2 often follows action1.
        """
        # Ensure vectors exist natively
        self.hd_space.generate_atomic_vector(action1)
        self.hd_space.generate_atomic_vector(action2)
        
        if action1 not in self.action_memory:
            self.action_memory[action1] = []
        if action2 not in self.action_memory[action1]:
            self.action_memory[action1].append(action2)
        
    def observe(self, event_vector: np.ndarray):
        """
        Update working memory (belief state) using a sliding window.
        This bounds the number of vectors bundled together, 
        preventing the Superposition Catastrophe (capacity collapse).
        """
        self.recent_events.append(event_vector)
        if len(self.recent_events) > self.belief_window:
            self.recent_events.pop(0) # Forget oldest event
            
        self.current_belief = self.hd_space.bundle(self.recent_events)

    def _tree_search(self, current_state_vec: np.ndarray, current_action: str, depth: int) -> float:
        """
        Recursive Deep Tree Search to calculate path integral of Expected Free Energy.
        Returns the best possible similarity score from this branch.
        """
        # Base case: max depth reached
        if depth == 0 or current_action not in self.action_memory or not self.action_memory[current_action]:
            return self.hd_space.similarity(current_state_vec, self.goal_vector)
            
        best_future_sim = -1.0
        
        for next_action in self.action_memory[current_action]:
            action_vec = self.hd_space.item_memory[next_action]
            # Simulate Next State = Bundle(Current State, Next Action)
            next_state_vec = self.hd_space.bundle([current_state_vec, action_vec])
            
            # Recurse
            branch_sim = self._tree_search(next_state_vec, next_action, depth - 1)
            if branch_sim > best_future_sim:
                best_future_sim = branch_sim
                
        # The EFE of taking `current_action` is the similarity achieved over the horizon
        return best_future_sim

    def plan(self):
        """
        HDC Sophisticated Inference via Deep Tree Search.
        """
        best_sim = -1.0
        best_action = None
        
        if not self.action_memory:
            return None
            
        # Evaluate root actions
        for action_label in self.action_memory.keys():
            action_vec = self.hd_space.item_memory[action_label]
            simulated_next_state = self.hd_space.bundle([self.current_belief, action_vec])
            
            # Launch tree search from this immediate action
            horizon_sim = self._tree_search(simulated_next_state, action_label, self.search_depth - 1)
            
            if horizon_sim > best_sim:
                best_sim = horizon_sim
                best_action = action_label
                
        return best_action, best_sim
