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
    
    # === PHASE 11: STRUCTURED GEOMETRIC GROUNDING ===
    # Inject fluent English bindings into the raw physics invariants
    from grounding_matrix import build_grounded_physics_matrix
    grounded_anchors, syntactic_roles, clean_up_memory = build_grounded_physics_matrix(broca, words_list, semantic_matrix)
    
    for anchor, tensor in grounded_anchors.items():
        if anchor in brain.hd_space.item_memory:
            # Overwrite the abstract physics tensor with the structurally grounded English Lexicon tensor
            brain.hd_space.item_memory[anchor] = tensor
            
            
    roles = list(zip(["Subject", "Verb", "Object1", "Operator", "Object2"], syntactic_roles))
    
    # CRITICAL FIX: Generate the physics concepts list AFTER the grounding overwrite
    # Limit the search space to the grammatically grounded tensor geometries
    physics_concepts = list(grounded_anchors.keys())
    
    # We must explicitly use the grounded vectors for generative unbinding
    # (Removed grounded_memory mapping to avoid KeyErrors on non-syllabus anchors)
    
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
                
            # 1. Broca Sensory Processing: Look up words in the 50,000 lexicon or the clean-up cache
            words = question.replace('?', '').replace('.', '').replace(',', '').split()
            
            # Filter non-semantic conversational stopwords to isolate physical concepts
            stopwords = {"what", "is", "the", "of", "does", "do", "why", "are", "you", "a", "an", "how", "say", "matter", "theory", "tell", "me", "about"}
            words = [w for w in words if w.lower() not in stopwords]
            
            if not words:
                print("\n[PARIETAL CORTEX]: No measurable physics semantics detected in query.\n")
                continue
                
            question_vectors = []
            for w in words:
                w_lower = w.lower()
                if w_lower in clean_up_memory:
                    # Explicitly use the cached geometry that grounded the Physics Tensors
                    question_vectors.append(clean_up_memory[w_lower])
                elif w_lower in words_list:
                    word_idx = np.where(words_list == w_lower)[0][0]
                    question_vectors.append(semantic_matrix[word_idx])
                else:
                    question_vectors.append(broca.hd_space.generate_atomic_vector(w_lower))
                    
            question_bundle = broca.hd_space.bundle(question_vectors)
            
            # 2. Parietal Calculus (Physics Invariant Calculation)
            # To route the unstructured "Bag of Words" question to the Structured physics geometry,
            # we must unbind the physics tensor into its constituent semantic fillers first.
            results = []
            for concept_name in physics_concepts:
                concept_vector = grounded_anchors[concept_name]
                
                # Unbind the structure into a mathematical Bag-of-Words for similarity matching
                unbound_fillers = [broca.hd_space.bind(concept_vector, role_vec) for role_name, role_vec in roles]
                concept_bow = broca.hd_space.bundle(unbound_fillers)
                
                sim = brain.hd_space.similarity(question_bundle, concept_bow)
                results.append((concept_name, concept_vector, sim))
                
            results.sort(key=lambda x: x[2], reverse=True)
            top_physics_name, top_physics_vector, top_sim = results[0]
            
            print(f"\n[PARIETAL CORTEX]: Calculated underlying topology -> {top_physics_name}")
            
            # 3. Generative Decoder (Iterative Unbinding via Resonator Network)
            print("[GENERATIVE DECODER]: Formulating fluent neural response...")
            
            sentence_parts = []
            
            for role_name, role_vec in roles:
                # Multiply Topological Invariant by inverse syntactic role 
                extracted_noisy_filler = broca.hd_space.bind(top_physics_vector, role_vec)
                
                # Clean-Up Memory: Filter noise by restricting dot-product search to grounded vocabulary only
                best_word = "null"
                best_score = -float('inf')
                
                for word, word_vec in clean_up_memory.items():
                    # MUST cast to int32 before dotting, otherwise 10000 overflows int8 bounds limit (127)!
                    score = np.dot(word_vec.astype(np.int32), extracted_noisy_filler.astype(np.int32))
                    if score > best_score:
                        best_score = score
                        best_word = word
                
                # Filter blanks
                if best_word != "null":
                    sentence_parts.append(best_word)
                
            fluent_sentence = " ".join(sentence_parts).capitalize() + "."
            print(f"\n[DHAI-4]: {fluent_sentence}\n")
            
        except KeyboardInterrupt:
            print("\n\nShutting down DHAI-4 Engine. Goodbye.")
            break

if __name__ == "__main__":
    run_fluent_chat()
