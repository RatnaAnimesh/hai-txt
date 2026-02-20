import numpy as np
from dhai4_hdc.utils.hdc_math import get_hd_space

class HDC_EventProcessor:
    def __init__(self, sim_threshold=0.05, dimension=10000):
        """
        Level 1: Event Processor (Wernicke).
        Implements Fast Structure Learning via HDC Bundling.
        """
        self.hd_space = get_hd_space(dimension)
        self.sim_threshold = sim_threshold
        
    def process_stream(self, vector_sequence: list[np.ndarray]) -> list[np.ndarray]:
        """
        Chunks a stream of item vectors into macroscopic 'Events' (Bundles).
        """
        if not vector_sequence:
            return []
            
        events = []
        current_chunk = [vector_sequence[0]]
        # The 'Event' is the bundled representation of the chunk
        current_bundle = vector_sequence[0]
        
        for i in range(1, len(vector_sequence)):
            curr_vec = vector_sequence[i]
            
            # Check similarity of incoming vector against the current context (bundle)
            sim = self.hd_space.similarity(curr_vec, current_bundle)
            
            # In massive dimensions, orthogonality is 0.0. 
            # Anything significantly above 0 is "similar".
            if sim < self.sim_threshold:
                # Surprise! The new vector is orthogonal/unrelated to the running context.
                # 1. Finalize the old bundle
                events.append(current_bundle)
                
                # 2. Start a new event chunk
                current_chunk = [curr_vec]
                current_bundle = curr_vec
            else:
                # 3. Add to chunk and update bundle
                current_chunk.append(curr_vec)
                current_bundle = self.hd_space.bundle(current_chunk)
                
        # Finalize last chunk
        if current_chunk:
            events.append(current_bundle)
            
        return events
