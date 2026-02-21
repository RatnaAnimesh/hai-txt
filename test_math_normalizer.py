import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dhai4_hdc.utils.fpe_math import get_fhrr_space

class LAtexGeometricNormalizer:
    """
    Commutative Geometric Isomorphism Normalizer.
    Parses a simplified Mathematical AST and deterministically sorts 
    commutative operators (like + and *) before FHRR binding, ensuring 
    that mathematically equivalent structures (A+B and B+A) 
    collapse to the exact same complex phasor geometry.
    """
    def __init__(self, dimension=10000):
        self.hd_space = get_fhrr_space(dimension)
        
        # Ground fundamental operators and operands
        self.r_add = self.hd_space.generate_atomic_vector("OP_ADD")
        self.r_mul = self.hd_space.generate_atomic_vector("OP_MUL")
        
        # Generate some constants for testing
        for var in ['A', 'B', 'C', 'V_GS', 'V_TH']:
             self.hd_space.generate_atomic_vector(f"VAR_{var}")

    def get_var_vector(self, var_name: str) -> np.ndarray:
        return self.hd_space.item_memory.get(f"VAR_{var_name}")

    def symmetric_bottom_up_bind(self, ast_node) -> np.ndarray:
        """
        Recursively traverses a Mathematical AST.
        If the node is a commutative operator, its children are evaluated,
        their resulting FHRR vectors are sorted deterministically 
        (e.g., by their complex phase sum or lexicographical name), 
        and then they are bound/bundled to guarantee symmetric invariance.
        """
        # Base Case: Leaf Node (Variable)
        if isinstance(ast_node, str):
            return self.get_var_vector(ast_node)
            
        # Recursive Case: Operator Node Dict {'op': '+', 'children': [node1, node2]}
        op = ast_node.get('op')
        children = ast_node.get('children', [])
        
        evaluated_children = [self.symmetric_bottom_up_bind(c) for c in children]
        
        if op in ['+', '*']:
            # Commutative Isomorphism Enforcement
            # To ensure A+B == B+A, we must sort the evaluated child vectors deterministically
            # before we bind them to the operator role.
            # We sort by the real-component sum of the complex vector 
            # (a deterministic, invariant property of the geometry itself).
            evaluated_children.sort(key=lambda v: np.sum(np.real(v)))
            
            # Now we perform the structural HDC operations
            # Bundle the children together symmetrically
            child_bundle = self.hd_space.bundle(evaluated_children)
            
            # Bind the bundle to the Operator role
            op_vec = self.r_add if op == '+' else self.r_mul
            return self.hd_space.bind(op_vec, child_bundle)
            
        else:
            raise ValueError(f"Unknown or non-commutative operator: {op}")

if __name__ == "__main__":
    print("Testing Commutative Geometric Isomorphism (LaTeX Normalizer)...")
    normalizer = LAtexGeometricNormalizer(10000)
    
    # Test 1: Simple Commutative Addition
    # Equation 1: V_GS + V_TH
    ast1 = {'op': '+', 'children': ['V_GS', 'V_TH']}
    
    # Equation 2: V_TH + V_GS (Algebraically identical, syntactically flipped)
    ast2 = {'op': '+', 'children': ['V_TH', 'V_GS']}
    
    vec1 = normalizer.symmetric_bottom_up_bind(ast1)
    vec2 = normalizer.symmetric_bottom_up_bind(ast2)
    
    sim12 = normalizer.hd_space.similarity(vec1, vec2)
    print(f"\nSimilarity between (V_GS + V_TH) and (V_TH + V_GS): {sim12:.6f}")
    if sim12 > 0.999:
        print("  [PASS] Commutative Isomorphism Achieved! The sequences collapsed to identical geometry.")
    
    # Test 2: Nested Commutative Operations
    # Eq A: (A + B) * C
    ast_A = {'op': '*', 'children': [
        {'op': '+', 'children': ['A', 'B']}, 
        'C'
    ]}
    
    # Eq B: C * (B + A)
    ast_B = {'op': '*', 'children': [
        'C',
        {'op': '+', 'children': ['B', 'A']}
    ]}
    
    vec_A = normalizer.symmetric_bottom_up_bind(ast_A)
    vec_B = normalizer.symmetric_bottom_up_bind(ast_B)
    
    sim_AB = normalizer.hd_space.similarity(vec_A, vec_B)
    print(f"\nSimilarity between (A + B) * C and C * (B + A): {sim_AB:.6f}")
    if sim_AB > 0.999:
        print("  [PASS] Deep Nested Commutative Isomorphism Achieved! Complete structural invariance.")
