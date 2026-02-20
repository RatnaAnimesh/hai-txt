import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
from dhai4_hdc.models.level1_wernicke import HDC_EventProcessor
from dhai4_hdc.models.level2_frontal import HDC_NarrativePlanner
from dhai4_hdc.utils.hdc_math import get_hd_space

def run_hdc_demo():
    print("=== DHAI-4 HDC: Vector Symbolic Architecture Demo ===\n")
    
    # Use standard dimension
    dim = 10000 
    
    print("--- Initializing Non-Connectionist Hierarchy ---")
    broca = HDC_SensoryInterface(dimension=dim)
    wernicke = HDC_EventProcessor(dimension=dim, sim_threshold=0.1) # Threshold to trigger chunking
    frontal = HDC_NarrativePlanner(dimension=dim)
    
    # 1. Setup Goal and Action Memory
    # Goal: A state representing "Happy" and "Ending"
    frontal.set_goal(["Happy", "Ending"])
    
    # Possible actions the agent can take (or propose)
    frontal.add_known_action("Fight")
    frontal.add_known_action("Run")
    frontal.add_known_action("Cry")
    frontal.add_known_action("Win")
    
    # 2. Simulated Sensory Stream
    print("\n--- Processing Sensory Stream (Words) ---")
    story_stream = [
        "The", "hero", "meets", "the", "monster",  # Event 1
        "The", "hero", "is", "hurt",               # Event 2
        "The", "hero", "finds", "strength"         # Event 3
    ]
    
    # Process sequentially
    vector_stream = []
    
    for word in story_stream:
        # Level 0 Encodes word natively into a 10,000d geometric space
        vec = broca.encode(word)
        vector_stream.append(vec)
        
    print(f"[Level 0] Encoded {len(story_stream)} words into HDC vectors.")
    
    # Level 1 Chunks into Events based on sequence similarity drop-offs
    events = wernicke.process_stream(vector_stream)
    print(f"[Level 1] Chunked stream into {len(events)} macroscopic HDC Events.")
    
    # Level 2 Observes the events and Plans
    for i, event_vec in enumerate(events):
        print(f"\n--- Processing Event {i+1} ---")
        frontal.observe(event_vec)
        frontal.plan()

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_hdc_demo()
