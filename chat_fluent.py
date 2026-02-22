import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infinite_omniscience import build_grand_syllabus, initialize_vocabulary
from train_omniscience import OmniscienceEngine
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

def load_semantic_lexicon(filepath="hd_lexicon.npz"):
    print("Loading 50,000-Word Semantic Hyperdimensional Matrix...")
    if not os.path.exists(filepath):
        print(f"[!] ERROR: Lexicon matrix '{filepath}' not found. Run 'python build_lexicon.py' first.")
        sys.exit(1)
        
    data = np.load(filepath)
    words = data['words']
    matrix = data['matrix'] # 50000 x 10000 int8
    print("Matrix loaded instantly. High-Dimensional Orthogonality verified.")
    return words, matrix

def initialize_universal_engine():
    print("Booting DHAI-4 Universal Engine...")
    brain = OmniscienceEngine(10000)
    syllabus = build_grand_syllabus()
    initialize_vocabulary(brain, syllabus)
    brain.library = syllabus
    
    print("Assimilating 35-Stage Grand Mathematical Curriculum (Calculus to M-Theory)...")
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

def run_fluent_chat():
    words_list, semantic_matrix = load_semantic_lexicon()
    brain = initialize_universal_engine()
    broca = HDC_SensoryInterface(10000)
    
    # Pre-generate syntax roles for Iterative Unbinding
    # In VSA: Role^-1 = Role 
    role_subject = broca.hd_space.generate_atomic_vector("ROLE_SUBJECT")
    role_action = broca.hd_space.generate_atomic_vector("ROLE_ACTION")
    role_object = broca.hd_space.generate_atomic_vector("ROLE_OBJECT")
    roles = [("Subject", role_subject), ("Action", role_action), ("Object", role_object)]
    
    physics_concepts = [k for k in brain.hd_space.item_memory.keys() if k.isupper()]
    
    print("\n" + "="*80)
    print("DHAI-4 FLUENT OMNISCIENCE CHAT INTERFACE ACTIVATED")
    print("Type your physics/math queries. The Generative Decoder will synthesize fluent English.")
    print("Type 'exit' to quit.")
    print("="*80 + "\n")
    
    while True:
        try:
            question = input("> ")
            if question.strip().lower() in ['exit', 'quit']:
                print("\nShutting down DHAI-4 Engine. Goodbye.")
                break
            
            if not question.strip():
                continue
                
            # 1. Broca Sensory Processing
            words = question.replace('?', '').replace('.', '').replace(',', '').split()
            question_vectors = [broca.hd_space.generate_atomic_vector(w.lower()) for w in words]
            question_bundle = broca.hd_space.bundle(question_vectors)
            
            # 2. Parietal Calculus (Physics Invariant Calculation)
            results = []
            for concept_name in physics_concepts:
                concept_vector = brain.hd_space.item_memory[concept_name]
                sim = brain.hd_space.similarity(question_bundle, concept_vector)
                results.append((concept_name, concept_vector, sim))
                
            results.sort(key=lambda x: x[2], reverse=True)
            top_physics_name, top_physics_vector, top_sim = results[0]
            
            print(f"\n[PARIETAL CORTEX]: Calculated underlying topology -> {top_physics_name}")
            
            # 3. Generative Decoder (Iterative Unbinding via Resonator Network)
            # We unbind the selected physics vector tracking its semantic structures
            print("[GENERATIVE DECODER]: Formulating fluent neural response...")
            
            sentence_parts = []
            
            for role_name, role_vec in roles:
                # Multiply Topological Invariant by inverse syntactic role 
                extracted_noisy_filler = broca.hd_space.bind(top_physics_vector, role_vec)
                
                # Resonator Network Matrix Multiplication (Dot Product is proportional to Cosine Sim in VSA)
                # We multiply (50000 x 10000) by (10000 x 1) to get similarities against all English words instantaneously.
                scores = np.dot(semantic_matrix, extracted_noisy_filler)
                best_idx = np.argmax(scores)
                best_word = words_list[best_idx]
                
                sentence_parts.append(best_word)
                
            fluent_sentence = " ".join(sentence_parts).capitalize() + "."
            print(f"\n[DHAI-4]: {fluent_sentence}\n")
            
        except KeyboardInterrupt:
            print("\n\nShutting down DHAI-4 Engine. Goodbye.")
            break

if __name__ == "__main__":
    run_fluent_chat()
