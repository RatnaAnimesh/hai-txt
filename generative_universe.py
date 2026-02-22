import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dhai4_hdc.models.level0_broca import HDC_SensoryInterface

def test_generative_decoder():
    print("="*80)
    print("DHAI-4 GENERATIVE DECODER (Iterative Unbinding)")
    print("="*80)
    
    dim = 10000
    broca = HDC_SensoryInterface(dim)
    
    # 1. Broca's Area: Generate Syntactic Roles
    # In VSA: Role^-1 = Role for bipolar vectors because x_i * x_i = (+1 or -1)^2 = +1
    role_subject = broca.hd_space.generate_atomic_vector("ROLE_SUBJECT")
    role_action = broca.hd_space.generate_atomic_vector("ROLE_ACTION")
    role_object = broca.hd_space.generate_atomic_vector("ROLE_OBJECT")
    
    # Generate Semantic Fillers (The Dictionary)
    semantic_dict = {
        "gravity": broca.hd_space.generate_atomic_vector("gravity"),
        "mass": broca.hd_space.generate_atomic_vector("mass"),
        "warps": broca.hd_space.generate_atomic_vector("warps"),
        "spacetime": broca.hd_space.generate_atomic_vector("spacetime"),
        "photon": broca.hd_space.generate_atomic_vector("photon"),
        "electron": broca.hd_space.generate_atomic_vector("electron")
    }
    
    # 2. The Engine calculates reality (The Parietal Invariant)
    # The agent "understands" that MASS WARPS SPACETIME
    bind_subj = broca.hd_space.bind(role_subject, semantic_dict["mass"])
    bind_act = broca.hd_space.bind(role_action, semantic_dict["warps"])
    bind_obj = broca.hd_space.bind(role_object, semantic_dict["spacetime"])
    
    # Superposition Bundle: The singular unreadable invariant token
    physics_invariant = broca.hd_space.bundle([bind_subj, bind_act, bind_obj])
    
    print("\n[PARIETAL CORTEX]: Topologically bound ('mass', 'warps', 'spacetime') into a single 10,000-D physical invariant.\n")
    
    # 3. Cognitive Decoupling: The Generative Decoder maps it back to English
    print("[GENERATIVE DECODER]: Iterative Unbinding Activated...")
    roles_to_probe = [("Subject", role_subject), ("Action", role_action), ("Object", role_object)]
    
    decoded_sentence = []
    
    for role_name, role_vec in roles_to_probe:
        # ** ITERATIVE UNBINDING ** 
        # Extracted_filler = physics_invariant (x) role_vec^-1
        # Because vector space is bipolar, role_vec == role_vec^-1
        extracted_noisy_filler = broca.hd_space.bind(physics_invariant, role_vec)
        
        # Snap the noisy extraction to the crisp semantic tokens via cosine similarity
        best_word = "<UNKNOWN>"
        best_sim = -1.0
        
        for word, word_vec in semantic_dict.items():
            sim = broca.hd_space.similarity(extracted_noisy_filler, word_vec)
            if sim > best_sim:
                best_sim = sim
                best_word = word
                
        decoded_sentence.append(best_word)
        print(f"  -> Extracted {role_name}: '{best_word}' (Cosine Confidence: {best_sim*100:.1f}%)")
        
    final_output = " ".join(decoded_sentence)
    print(f"\n[DHAI-4 OUTPUT TEXT]: {final_output.capitalize()}.")

if __name__ == "__main__":
    test_generative_decoder()
