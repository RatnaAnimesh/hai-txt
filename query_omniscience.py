import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_omniscience import OmniscienceEngine
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

class DHAI4_QueryEngine:
    def __init__(self):
        print("Booting DHAI-4 and training on Master Curriculum...")
        # 1. Train the engine to Omniscience
        self.brain = OmniscienceEngine(10000)
        
        # We silently run the training loop to build the geometric knowledge base
        while self.brain.library:
            best_candidate = None
            best_efe = -1.0
            for book in self.brain.library:
                vecs = [self.brain.hd_space.item_memory[v] for v in book['vecs']]
                efe = self.brain.evaluate_efe(vecs)
                if 0.05 <= efe <= 0.85 and efe > best_efe:
                    best_efe = efe
                    best_candidate = book
            if best_candidate:
                for v_name in best_candidate['vecs']:
                    self.brain.knowledge_vectors.append(self.brain.hd_space.item_memory[v_name])
                self.brain.current_kb = self.brain.hd_space.bundle(self.brain.knowledge_vectors)
                self.brain.library.remove(best_candidate)
            else:
                break
                
        # The library of concepts the brain now logically understands
        self.learned_concepts = list(self.brain.hd_space.item_memory.keys())
        
        # 2. Attach the Broca linguistic interface
        self.broca = HDC_SensoryInterface(10000)
        
    def query(self, question: str):
        print(f"\n[USER]: {question}")
        
        # Convert the natural language question into a hyperdimensional semantic topology
        # We split the words and generate a rapid bundle
        words = question.replace('?', '').split()
        question_vectors = []
        for word in words:
            # We use semantic hashing to assign the word a vector
            vec = self.broca.hd_space.generate_atomic_vector(word.lower())
            question_vectors.append(vec)
            
        # The geometric "thought" of the question
        question_bundle = self.broca.hd_space.bundle(question_vectors)
        
        # Now, DHAI-4 searches its Parietal/Wernicke knowledge base for the most 
        # geometrically resonant physics/math concepts to answer the question
        print("[DHAI-4]: Processing geometric resonance across physics matrix...")
        
        results = []
        for concept_name in self.learned_concepts:
            # Skip the random words we just hashed for the question
            if concept_name.islower():
                continue
                
            concept_vector = self.brain.hd_space.item_memory[concept_name]
            
            # How closely does the question's geometry align with the physical law's geometry?
            sim = self.brain.hd_space.similarity(question_bundle, concept_vector)
            results.append((concept_name, sim))
            
        # Sort by highest topological similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("  -> Top Activated Conceptual Bundles:")
        for concept, sim in results[:3]:
            # Scale the similarity for readable output confidence
            confidence = max(0, sim * 100 * 5) # Heuristic scaling for demonstration
            print(f"     * {concept} (Resonance: {confidence:.1f}%)")

if __name__ == "__main__":
    bot = DHAI4_QueryEngine()
    print("\n" + "="*70)
    print("DHAI-4 TERMINAL CHAT INTERFACE ACTIVATED")
    print("="*70)
    
    # I am asking these 3 questions on the user's behalf to test the geometry:
    
    bot.query("How do electrical dynamics relate to continuous calculus?")
    
    bot.query("What are the foundational limits of a margin call in finance?")
    
    bot.query("Can we unify quantum mechanics and gravity?")
    
    print("\n" + "="*70)
