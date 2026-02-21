import numpy as np

class FHRRSpace:
    """
    Fourier Holographic Reduced Representations (FHRR) Space.
    Instead of binary or bipolar integers, dimensions are represented as
    complex numbers (phasors) on the unit circle: v_k = e^{i \theta_k}.
    This is required to geometrically support Fractional Power Encoding (FPE)
    so the architecture can "feel" continuous scalar proximity.
    """
    def __init__(self, dimension=10000):
        self.dimension = dimension
        self.item_memory = {} # Maps string labels to their FHRR base vectors

    def generate_atomic_vector(self, name: str) -> np.ndarray:
        """
        Generates a random complex vector uniformly distributed on the unit circle.
        """
        if name in self.item_memory:
            return self.item_memory[name]
            
        # Random angles between -pi and pi
        angles = np.random.uniform(-np.pi, np.pi, self.dimension)
        # Convert to complex plane points on the unit circle
        vec = np.exp(1j * angles)
        
        self.item_memory[name] = vec
        return vec

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        FHRR Binding (Multiplication). Pointwise multiplication of complex vectors,
        which equates to the addition of their phase angles.
        """
        return v1 * v2

    def inverse(self, v: np.ndarray) -> np.ndarray:
        """
        FHRR Inverse Binding. The exact mathematical inverse is the complex conjugate.
        Binding a state with its conjugate yields the identity vector (all 1s).
        """
        return np.conj(v)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        FHRR Bundling (Addition). Pointwise addition of complex vectors,
        followed by projection back strictly onto the unit circle.
        """
        if not vectors:
            return np.ones(self.dimension, dtype=np.complex128) # Default identity
            
        summed = np.sum(vectors, axis=0)
        magnitudes = np.abs(summed)
        
        # Avoid division by zero for destructive interference
        safe_mags = np.where(magnitudes == 0, 1.0, magnitudes)
        
        # Project back to unit circle
        return summed / safe_mags

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        FHRR Similarity. Calculates the mean cosine of the phase angle differences.
        Mathematically equivalent to the average real part of v1 * conjugate(v2).
        """
        # Element-wise product of v1 and the inverse (conjugate) of v2
        phase_diffs = v1 * self.inverse(v2)
        # The real part of e^{i\theta} is cos(\theta)
        return float(np.mean(np.real(phase_diffs)))

    def fractionally_encode(self, base_vec: np.ndarray, scalar: float) -> np.ndarray:
        """
        Fractional Power Encoding (FPE).
        Encodes a continuous scalar natively into geometry by raising the base vector
        to the power of the scalar natively via phase multiplication: V(x) = B^x.
        Because B = e^{i\theta}, B^x = e^{i x \theta}. 
        """
        # Extract original angles using arctan2(imag, real)
        angles = np.angle(base_vec)
        # Scale angles by the scalar
        scaled_angles = angles * scalar
        # Project back to complex phasors
        return np.exp(1j * scaled_angles)

    def extract_delta(self, current_state: np.ndarray, previous_state: np.ndarray) -> np.ndarray:
        """
        Geometric Calculus: Differentiation.
        Extracts the explicit mathematical transformation delta that occurred 
        between the previous and current state frames by binding with the inverse.
        """
        return self.bind(current_state, self.inverse(previous_state))


# Global singleton helper
_FHRR_SPACE = None

def get_fhrr_space(dimension=10000) -> FHRRSpace:
    global _FHRR_SPACE
    if _FHRR_SPACE is None or _FHRR_SPACE.dimension != dimension:
        _FHRR_SPACE = FHRRSpace(dimension)
    return _FHRR_SPACE
