import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
from chat_fluent import load_semantic_lexicon

def build_grounded_physics_matrix(broca, words_list, semantic_matrix):
    print("="*80)
    print("DHAI-4 STRUCTURED GEOMETRIC GROUNDING")
    print("="*80)
    
    # Establish the grammatical structure (The "Chomsky Fix")
    role_subject = broca.hd_space.generate_atomic_vector("ROLE_SUBJECT")
    role_verb = broca.hd_space.generate_atomic_vector("ROLE_ACTION")
    role_object1 = broca.hd_space.generate_atomic_vector("ROLE_OBJECT")
    role_operator = broca.hd_space.generate_atomic_vector("ROLE_OPERATOR")
    role_object2 = broca.hd_space.generate_atomic_vector("ROLE_OBJECT2")
    
    syntactic_roles = [role_subject, role_verb, role_object1, role_operator, role_object2]
    
    glossary_path = "physics_glossary.txt"
    grounded_anchors = {}
    clean_up_memory = {}
    
    print("\n[GROUNDING PIPELINE]: Extracting words from Lexicon and binding to Syntax Roles...")
    
    with open(glossary_path, "r") as f:
        for line in f:
            if "|" not in line: continue
            
            anchor_name, sentence = line.strip().split("|")
            words = sentence.replace(".", "").lower().split()
            
            # Pad or truncate sentence to exactly 5 tokens for our fixed role array demonstration
            while len(words) < 5:
                words.append("null")
            words = words[:5]
            
            bound_tensors = []
            
            for word, role in zip(words, syntactic_roles):
                # 1. Look up the exact orthogonal vector from the massive 50K dictionary or Cache
                if word in clean_up_memory:
                    word_vector = clean_up_memory[word]
                elif word in words_list:
                    word_idx = np.where(words_list == word)[0][0]
                    word_vector = semantic_matrix[word_idx]
                else:
                    # If word isn't in top 50k and not cached, generate a new persistent random geometry
                    word_vector = np.random.choice([-1, 1], size=10000).astype(np.int8)
                
                clean_up_memory[word] = word_vector
                
                # 2. Structural Binding (Filler (x) Role) to prevent "Bag of Words" caveman syntax
                bound_word = broca.hd_space.bind(role, word_vector)
                bound_tensors.append(bound_word)
                
            # 3. Superposition Bundling (Construct the definitive mathematical concept)
            invariant_physics_topology = broca.hd_space.bundle(bound_tensors)
            grounded_anchors[anchor_name] = invariant_physics_topology
            
            print(f"  -> Grounded Semantic Syntax: [{anchor_name}]")
            
    print(f"\n[GROUNDING COMPLETE] {len(grounded_anchors)} distinct Physics dimensions have been structurally bridged to the 50,000-Word Lexicon.")
    return grounded_anchors, syntactic_roles, clean_up_memory

if __name__ == "__main__":
    words_list, semantic_matrix = load_semantic_lexicon()
    broca = HDC_SensoryInterface(10000)
    build_grounded_physics_matrix(broca, words_list, semantic_matrix)
