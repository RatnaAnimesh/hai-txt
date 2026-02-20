import numpy as np

class HDSpace:
    def __init__(self, dimension=10000):
        """
        Hyperdimensional Vector Space (Vector Symbolic Architecture).
        Vectors are bipolar (+1, -1).
        """
        self.d = dimension
        # Item memory stores the atomic base vectors (e.g., words, letters)
        self.item_memory = {}
        
    def generate_atomic_vector(self, label: str) -> np.ndarray:
        """
        Generate a random bipolar vector (+1 or -1) for a new atomic concept.
        Because d is huge (e.g. 10000), any two random vectors are orthogonal.
        """
        if label in self.item_memory:
            return self.item_memory[label]
            
        # Generate random +1 or -1
        vec = np.random.choice([1, -1], size=self.d).astype(np.int8)
        self.item_memory[label] = vec
        return vec

    @staticmethod
    def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Binding: Element-wise multiplication.
        Result is orthogonal to both v1 and v2.
        Acts like a variable assignment: Role(x)Filler.
        """
        return v1 * v2

    @staticmethod
    def bundle(vectors: list[np.ndarray]) -> np.ndarray:
        """
        Bundling: Addition followed by thresholding.
        Result is similar (high cosine similarity) to all input vectors.
        Acts like a Set or a Cluster.
        """
        if not vectors:
            raise ValueError("Cannot bundle an empty list of vectors.")
            
        # Sum the vectors
        sum_vec = np.sum(vectors, axis=0)
        
        # Threshold: if > 0 -> +1, if < 0 -> -1. If 0 -> pick randomly (tie breaker)
        bundled_vec = np.where(sum_vec > 0, 1, -1).astype(np.int8)
        
        # Tie breakers
        zero_indices = np.where(sum_vec == 0)[0]
        if len(zero_indices) > 0:
            bundled_vec[zero_indices] = np.random.choice([1, -1], size=len(zero_indices))
            
        return bundled_vec

    @staticmethod
    def permute(v: np.ndarray, shifts=1) -> np.ndarray:
        """
        Permutation: Cyclical shift.
        Used to encode sequence/time.
        Result is orthogonal to original vector.
        """
        return np.roll(v, shifts)

    @staticmethod
    def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Cosine similarity between two bipolar vectors.
        For +1/-1 vectors, this is mathematically equivalent to:
        (1 - 2 * Hamming_Distance / d)
        
        Returns:
            float: 1.0 (identical) down to -1.0 (opposite). 
                   0.0 implies orthogonality.
        """
        # Dot product divided by magnitudes.
        # Since vectors are +1/-1, magnitude is always sqrt(d)
        # So dot product / d is the exact cosine similarity.
        dot_product = np.dot(v1, v2)
        return float(dot_product) / len(v1)

# Singleton helper
_global_space = None

def get_hd_space(dimension=10000):
    global _global_space
    if _global_space is None:
        _global_space = HDSpace(dimension)
    return _global_space
