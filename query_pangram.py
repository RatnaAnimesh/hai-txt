import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_omniscience import OmniscienceEngine
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

def ask_dhai4():
    print("Booting DHAI-4...")
    brain = OmniscienceEngine(10000)
    
    # Train
    while brain.library:
        best_candidate = None
        best_efe = -1.0
        for book in brain.library:
            vecs = [brain.hd_space.item_memory[v] for v in book['vecs']]
            efe = brain.evaluate_efe(vecs)
            if 0.05 <= efe <= 0.85 and efe > best_efe:
                best_efe = efe
                best_candidate = book
        if best_candidate:
            for v_name in best_candidate['vecs']:
                brain.knowledge_vectors.append(brain.hd_space.item_memory[v_name])
            brain.current_kb = brain.hd_space.bundle(brain.knowledge_vectors)
            brain.library.remove(best_candidate)
        else:
            break
            
    broca = HDC_SensoryInterface(10000)
    learned_concepts = list(brain.hd_space.item_memory.keys())
    
    question = "Give me a sentence that covers all the 26 alphabets in it"
    print(f"\n[USER]: {question}")
    
    words = question.replace('?', '').split()
    question_vectors = []
    for word in words:
        vec = broca.hd_space.generate_atomic_vector(word.lower())
        question_vectors.append(vec)
        
    question_bundle = broca.hd_space.bundle(question_vectors)
    print("[DHAI-4]: Processing geometric resonance across physics matrix...\n")
    
    results = []
    for concept_name in learned_concepts:
        if concept_name.islower(): continue
        concept_vector = brain.hd_space.item_memory[concept_name]
        sim = brain.hd_space.similarity(question_bundle, concept_vector)
        results.append((concept_name, sim))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("  -> Top Activated Conceptual Bundles:")
    for concept, sim in results[:3]:
        confidence = max(0, sim * 100 * 5)
        print(f"     * {concept} (Resonance: {confidence:.1f}%)")

if __name__ == "__main__":
    ask_dhai4()
