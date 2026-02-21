import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infinite_omniscience import build_grand_syllabus, initialize_vocabulary
from train_omniscience import OmniscienceEngine
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

def initialize_universal_engine():
    """
    Silently boots the engine, initializes the 35-stage syllabus, and 
    assimilates all orthogonal physics dimensions into the hyperdimensional space.
    """
    print("Booting DHAI-4 Universal Hyperdimensional Matrix...")
    brain = OmniscienceEngine(10000)
    syllabus = build_grand_syllabus()
    
    initialize_vocabulary(brain, syllabus)
    brain.library = syllabus
    
    print("\nAssimilating 35-Stage Grand Curriculum (Calculus to M-Theory)...")
    
    # Fast-track assimilation without terminal readouts
    while brain.library:
        best_candidate = None
        best_efe = -1.0
        
        for book in brain.library:
            vec_list = [brain.hd_space.item_memory[v] for v in book['vecs']]
            efe = brain.evaluate_efe(vec_list)
            
            if 0.05 <= efe <= 0.95:
                if efe > best_efe:
                    best_efe = efe
                    best_candidate = book
                    
        if best_candidate:
            for v_name in best_candidate['vecs']:
                already_known = False
                incoming_vec = brain.hd_space.item_memory[v_name]
                for known_vec in brain.knowledge_vectors:
                    if (incoming_vec == known_vec).all():
                        already_known = True
                        break
                
                if not already_known:
                    brain.knowledge_vectors.append(incoming_vec)
                    
            brain.current_kb = brain.hd_space.bundle(brain.knowledge_vectors)
            brain.library.remove(best_candidate)
        else:
            print("\n[!] FATAL DEADLOCK DURING SILENT BOOT [!]")
            sys.exit(1)
            
    print(f"Assimilation Complete: {len(brain.knowledge_vectors)} Orthogonal Physical Dimensions Anchored.")
    return brain

def run_live_terminal_chat():
    brain = initialize_universal_engine()
    
    # Boot the Linguistic Sensory Interface
    broca = HDC_SensoryInterface(10000)
    
    # We only want to search across the actual theoretical physics variables,
    # not random English words the user types the first time
    physics_concepts = [k for k in brain.hd_space.item_memory.keys() if k.isupper()]
    
    print("\n" + "="*80)
    print("DHAI-4 OMNISCIENCE CHAT INTERFACE ACTIVATED")
    print("Type your physics/math queries natively in English. Type 'exit' to quit.")
    print("="*80 + "\n")
    
    while True:
        try:
            # Live, interactive prompt for the user
            question = input("> ")
            if question.strip().lower() in ['exit', 'quit']:
                print("\nShutting down DHAI-4 Engine. Goodbye.")
                break
            
            if not question.strip():
                continue
                
            # Broca hashes the English into a topological representation
            words = question.replace('?', '').replace('.', '').replace(',', '').split()
            question_vectors = []
            for word in words:
                vec = broca.hd_space.generate_atomic_vector(word.lower())
                question_vectors.append(vec)
                
            question_bundle = broca.hd_space.bundle(question_vectors)
            
            # The Frontal Cortex queries the assimilated Physics Matrix
            results = []
            for concept_name in physics_concepts:
                concept_vector = brain.hd_space.item_memory[concept_name]
                sim = brain.hd_space.similarity(question_bundle, concept_vector)
                results.append((concept_name, sim))
                
            # Sort by highest topological resonance
            results.sort(key=lambda x: x[1], reverse=True)
            
            print("\n[DHAI-4]: Activating Universal Concept Geometry...")
            for concept, sim in results[:4]: # Show top 4 resonant concepts
                # Scale the raw cosine similarity into a readable confidence percentage
                confidence = max(0, sim * 100 * 5)
                print(f"  * {concept} (Resonance: {confidence:.1f}%)")
            print() # Blank line for spacing
            
        except KeyboardInterrupt:
            print("\n\nShutting down DHAI-4 Engine. Goodbye.")
            break

if __name__ == "__main__":
    run_live_terminal_chat()
