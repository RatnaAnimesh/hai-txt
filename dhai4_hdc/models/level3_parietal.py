import numpy as np
from dhai4_hdc.utils.fpe_math import get_fhrr_space

class HDC_ParietalCortex:
    """
    Level 3: Parietal Cortex
    Grounded Mathematical & Physical Understanding Engine.
    Operates strictly via continuous Fourier Holographic Reduced Representations (FHRR).
    No discrete Python math parsing; all transitions are geometric phase shifts.
    """
    def __init__(self, dimension=10000):
        self.hd_space = get_fhrr_space(dimension)
        
        # Grounded semantic roots for physical properties
        self.r_mass = self.hd_space.generate_atomic_vector("ROLE_MASS")
        self.r_velocity = self.hd_space.generate_atomic_vector("ROLE_VELOCITY")
        self.r_energy = self.hd_space.generate_atomic_vector("ROLE_ENERGY")
        
        # Universal Physics Base Vectors (The B in B^x)
        self.scalar_base = self.hd_space.generate_atomic_vector("SCALAR_BASE")

    def encode_physical_state(self, mass: float, velocity: float) -> np.ndarray:
        """
        Creates a "World State Graph" by fractionally encoding continuous
        magnitudes and binding them to structural property roles.
        """
        v_mass = self.hd_space.fractionally_encode(self.scalar_base, mass)
        v_vel = self.hd_space.fractionally_encode(self.scalar_base, velocity)
        
        bound_mass = self.hd_space.bind(self.r_mass, v_mass)
        bound_vel = self.hd_space.bind(self.r_velocity, v_vel)
        
        return self.hd_space.bundle([bound_mass, bound_vel])

    def encode_kinetic_energy(self, mass: float, velocity: float) -> np.ndarray:
        """
        Internally calculates KE = 0.5 * m * v^2 natively through fractional geometry.
        """
        energy = 0.5 * mass * (velocity ** 2)
        v_energy = self.hd_space.fractionally_encode(self.scalar_base, energy)
        return self.hd_space.bind(self.r_energy, v_energy)

    def assess_transition_physics(self, state1_m: float, state1_v: float, 
                                  state2_m: float, state2_v: float) -> float:
        """
        The Absolute Geometric Prior: Conservation of Energy.
        If this spikes, the transition is physically impossible.
        """
        prior_energy_vec = self.encode_kinetic_energy(state1_m, state1_v)
        predicted_energy_vec = self.encode_kinetic_energy(state2_m, state2_v)
        
        sim = self.hd_space.similarity(prior_energy_vec, predicted_energy_vec)
        return 1.0 - max(0, sim)
        
    def assess_contextual_prior(self, current_state_vec: np.ndarray, 
                                contextual_boundary_vec: np.ndarray) -> float:
        """
        The Contextual Geometric Prior: Institutional/Environmental Rules.
        Evaluates boundary conditions (e.g. Margin Calls) that do not break physics
        but dictate severe environmental penalties.
        """
        sim = self.hd_space.similarity(current_state_vec, contextual_boundary_vec)
        return 1.0 - sim

    def measurement_operator(self, continuous_vec: np.ndarray, 
                             discrete_vocabulary: dict) -> str:
        """
        Continuous-to-Symbolic Binding.
        Collapses a continuous FHRR mathematical state into the nearest 
        discrete semantic token for Frontal Cortex narrative planning.
        """
        best_match = "<UNKNOWN>"
        highest_sim = -1.0
        
        for word, word_vec in discrete_vocabulary.items():
            sim = self.hd_space.similarity(continuous_vec, word_vec)
            if sim > highest_sim:
                highest_sim = sim
                best_match = word
                
        return best_match
