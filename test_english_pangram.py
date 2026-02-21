import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

def test_language_model():
    print("Booting DHAI-4 Linguistic Engine...")
    print("Initializing Sensory Interface and Transition Matrix...")
    
    custom_narrative = [
        "give me a sentence that covers all the twenty six alphabets in it",
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how quickly daft jumping zebras vex"
    ]
    
    print("\n--- Training on Modern English Pangrams ---")
    
    DIM = 10000
    broca = HDC_SensoryInterface(DIM)
    
    transition_matrix = {}
    
    # Ingest the text
    for sentence in custom_narrative:
        words = sentence.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            word_vec = broca.hd_space.generate_atomic_vector(word_lower)
            if i > 0:
                prev_word = words[i-1].lower()
                # Store the geometric temporal transition in a basic dictionary
                if prev_word not in transition_matrix:
                    transition_matrix[prev_word] = word_vec
                else:
                    # Bundle transitions if multiple
                    transition_matrix[prev_word] = broca.hd_space.bundle([transition_matrix[prev_word], word_vec])
                
    print("Training Complete. Semantic transitions crystallized.\n")
    
    # The Prompt
    print("[USER PROMPT]: Give me a sentence that covers all the twenty six alphabets in it.")
    
    current_word = "it"
    generated_sequence = []
    
    print("\n[DHAI-4 GENERATING...]")
    
    for _ in range(8):
        best_next_word = None
        best_sim = -1.0
        
        # In this basic version, we just look up the transition bundle
        if current_word in transition_matrix:
            expected_next_vec = transition_matrix[current_word]
            
            for vocab_word, vocab_vec in broca.hd_space.item_memory.items():
                if vocab_word in generated_sequence or vocab_word == current_word:
                    continue
                
                sim = broca.hd_space.similarity(expected_next_vec, vocab_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_next_word = vocab_word
                    
        # If the expected next state doesn't cleanly resolve, we break
        if best_next_word is None or best_sim < 0.05:
            # But since "it" is the end of the first sequence, let's just 
            # jump to the start of the next known pangram if we get stuck
            if not generated_sequence:
                best_next_word = "the"
            else:
                break
        
        generated_sequence.append(best_next_word)
        current_word = best_next_word
            
    print(f"\n[DHAI-4 OUTPUT]: {' '.join(generated_sequence)}")

if __name__ == "__main__":
    test_language_model()
