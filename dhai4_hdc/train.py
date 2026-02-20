import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
from dhai4_hdc.models.level1_wernicke import HDC_EventProcessor
from dhai4_hdc.models.level2_frontal import HDC_NarrativePlanner
from dhai4_hdc.utils.data_loader import CorpusLoader

def run_training_orchestrator():
    print("=== DHAI-4 HDC: Full Architecture Setup & Training Loop ===\n")
    
    dim = 10000 
    
    print("--- Initializing Hierarchy ---")
    broca = HDC_SensoryInterface(dimension=dim)
    wernicke = HDC_EventProcessor(dimension=dim, sim_threshold=0.1) 
    frontal = HDC_NarrativePlanner(dimension=dim, search_depth=3)
    
    # 1. Setup Planner Search Space (A synthetic causal graph for testing Deep Search)
    frontal.set_goal(["Hero", "Victory"])
    frontal.learn_action_transition("Fight", "Win")
    frontal.learn_action_transition("Fight", "Lose")
    frontal.learn_action_transition("Lose", "Run")
    frontal.learn_action_transition("Win", "Celebrate")
    frontal.learn_action_transition("Run", "Hide")
    frontal.learn_action_transition("Hide", "Rest")

    # 2. Setup Data Loader
    loader = CorpusLoader()
    # A synthetic "streaming" corpus representing environment observations
    synthetic_corpus = "The hero meets the monster . The hero is hurt ! The hero finds courage . He strikes the monster . The monster falls ."
    stream = loader.stream_tokens(text_content=synthetic_corpus)
    
    print("\n--- Simulating Simultaneous Multi-Level Training ---")
    
    prev_l0_bound = None
    running_l1_sequence = []
    
    # Unified Training Loop
    for token in stream:
        # --- LEVEL 0 Processing ---
        # 1. Bind Role and Filler
        l0_out = broca.encode(token)
        curr_bound = l0_out["bound_state"]
        
        # 2. Hebbian Sequence Learning at L0 (Vector B -> Vector C)
        if prev_l0_bound is not None:
            broca.learn_transition(prev_l0_bound, curr_bound)
        prev_l0_bound = curr_bound
        
        # Feed up to Level 1
        running_l1_sequence.append(curr_bound)
        
        # --- LEVEL 1 Processing ---
        # 3. Check for Event Chunks natively
        events = wernicke.process_stream(running_l1_sequence)
        
        if events:
            # An event threshold was crossed!
            new_event_vector = events[-1] # the most recently completed bundle
            
            # --- LEVEL 2 Processing ---
            # 4. Update Beliefs with new Macroscopic Event
            frontal.observe(new_event_vector)
            
            # 5. Plan (Deep Tree Search checking N-steps ahead)
            print(f"\n[L1 Event Discovered] Processing token window ending at '{token}'")
            best_action, score = frontal.plan()
            print(f"[L2 Deep Search] Selected Plan: {best_action} (Long-Horizon Sim: {score:.4f})")
            
            # Clear the local running sequence because the event is finalized
            running_l1_sequence = [curr_bound]

    print(f"\n[Training Stats]")
    print(f"L0 Graph Transitions Learned: {len(broca.transition_counts)}")
    print("=== Training Complete ===")

if __name__ == "__main__":
    run_training_orchestrator()
