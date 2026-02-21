import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dhai4_hdc.utils.fpe_math import get_fhrr_space

class EpistemicForager:
    """
    Autonomous Epistemic Forager (Curriculum Learning)
    Evaluates the Expected Free Energy (EFE) of potential textbook/paper abstracts.
    Selects texts in the "Zone of Proximal Development":
      - Rejects trivial texts (EFE ~ 0.0) -> No information gain
      - Rejects incomprehensible texts (EFE ~ 1.0) -> Unanchored phase noise
      - Selects texts with moderate EFE -> Manageable geometric resolution
    """
    def __init__(self, dimension=10000):
        self.hd_space = get_fhrr_space(dimension)
        
        # The Current Geometry: What the agent already mathematically understands
        # We simulate this by generating stable established bundles.
        self.current_knowledge_base = self.hd_space.bundle([
            self.hd_space.generate_atomic_vector("MATH_EQUALITY"),
            self.hd_space.generate_atomic_vector("MATH_ADDITION"),
            self.hd_space.generate_atomic_vector("PHYSICS_MASS"),
            self.hd_space.generate_atomic_vector("PHYSICS_VELOCITY")
        ])

    def evaluate_abstract(self, abstract_vectors: list[np.ndarray], title: str) -> float:
        """
        Simulates the Frontal Cortex reading an abstract.
        Calculates how the abstract's semantic geometry interacts with the current knowledge base.
        """
        # Bundle the abstract into a single geometric impression
        abstract_geometry = self.hd_space.bundle(abstract_vectors)
        
        # Calculate Epistemic Value: How much does this challenge the current knowledge?
        # A perfectly understood text will have high similarity to the knowledge base (EFE ~ 0).
        # A completely alien text will have orthogonal/zero similarity (EFE ~ 1).
        sim = self.hd_space.similarity(self.current_knowledge_base, abstract_geometry)
        
        # EFE = Surprise / Divergence
        efe = 1.0 - max(0, sim)
        return efe

    def select_curriculum(self, candidate_texts: list[dict]) -> dict:
        """
        Filters candidates for the Zone of Proximal Development.
        """
        best_candidate = None
        optimal_epistemic_value = -1.0
        
        print("\n--- Generating Epistemic Value (EFE) for Candidates ---")
        for candidate in candidate_texts:
            title = candidate['title']
            efe = self.evaluate_abstract(candidate['vectors'], title)
            
            # The Zone of Proximal Development Filter
            # EFE < 0.1: Too trivial. I already know this. (e.g. 1+1=2)
            # EFE > 0.9: Too complex/alien. Pure noise. (e.g. unanchored tensor calculus)
            # We want the text that maximizes EFE *within* the readable bounds.
            if efe < 0.1:
                status = f"REJECTED - TRIVIAL (EFE: {efe:.4f})"
            elif efe > 0.9:
                status = f"REJECTED - INCOMPREHENSIBLE (EFE: {efe:.4f})"
            else:
                status = f"VALID PROXIMAL ZONE (EFE: {efe:.4f})"
                if efe > optimal_epistemic_value:
                    optimal_epistemic_value = efe
                    best_candidate = candidate
                    
            print(f"[{title:<40}] -> {status}")
            
        return best_candidate

if __name__ == "__main__":
    print("Initializing DHAI-4 Autonomous Epistemic Forager...")
    forager = EpistemicForager(10000)
    
    # Simulate mathematical text embedding interactions
    v_eq = forager.hd_space.item_memory["MATH_EQUALITY"]
    v_add = forager.hd_space.item_memory["MATH_ADDITION"]
    v_mass = forager.hd_space.item_memory["PHYSICS_MASS"]
    v_vel = forager.hd_space.item_memory["PHYSICS_VELOCITY"]
    
    # 1. Trivial Text: "1 plus 1 equals 2"
    vectors_trivial = [v_eq, v_add, v_eq, v_add]
    
    # 2. Proximal Text: "Kinetic Energy involves mass and velocity in a dynamic equation"
    # Mixes known concepts with a few new ones (unanchored noise)
    v_new1 = forager.hd_space.generate_atomic_vector("NOVEL_KINETIC")
    v_new2 = forager.hd_space.generate_atomic_vector("NOVEL_DYNAMIC")
    vectors_proximal = [v_mass, v_vel, v_eq, v_new1, v_new2]
    
    # 3. Incomprehensible Text: "The Ricci tensor dictates geodetic curvature in a non-Euclidean manifold"
    # Entirely unanchored vectors
    vectors_alien = [
        forager.hd_space.generate_atomic_vector("ALIEN_RICCI"),
        forager.hd_space.generate_atomic_vector("ALIEN_TENSOR"),
        forager.hd_space.generate_atomic_vector("ALIEN_MANIFOLD"),
        forager.hd_space.generate_atomic_vector("ALIEN_GEODETIC")
    ]
    
    candidates = [
        {"title": "Basic Arithmetic for Toddlers", "vectors": vectors_trivial},
        {"title": "Introduction to Classical Mechanics", "vectors": vectors_proximal},
        {"title": "Advanced General Relativity (Gr-Qc)", "vectors": vectors_alien}
    ]
    
    best = forager.select_curriculum(candidates)
    
    print("\n" + "="*70)
    if best:
        print(f"FORAGING DECISION: DHAI-4 dynamically selected '{best['title']}'")
        print("Reasoning: This text provides maximum Epistemic Value while remaining geometrically anchored.")
    else:
        print("FORAGING DECISION: No texts found in the Zone of Proximal Development.")
    print("="*70)
