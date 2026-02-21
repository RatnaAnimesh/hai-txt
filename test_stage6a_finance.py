import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dhai4_hdc.utils.fpe_math import get_fhrr_space
from dhai4_hdc.models.level3_parietal import HDC_ParietalCortex

class MarginCallAssessor:
    """
    Stage 6A Benchmark: Quantitative Finance & Continuous Pricing
    Uses FHRR geometry to model continuous futures contracts.
    Enforces maintenance margin requirements geometrically via Expected Free Energy spikes.
    """
    def __init__(self, dimension=10000):
        self.hd_space = get_fhrr_space(dimension)
        self.parietal = HDC_ParietalCortex(dimension)
        
        # Grounding vectors
        self.r_account = self.hd_space.generate_atomic_vector("ROLE_ACCOUNT_BALANCE")
        
        # The universal FPE scalar base
        self.scalar_base = self.parietal.scalar_base
        
        # Stage 6A Contextual Prior (C_contextual): Maintenance Margin
        self.maintenance_margin = 5000.0
        v_margin = self.hd_space.fractionally_encode(self.scalar_base, self.maintenance_margin)
        self.prior_margin_state = self.hd_space.bind(self.r_account, v_margin)
        
        # Discrete Semantic Vocabulary for the Measurement Operator
        self.semantic_vocab = {
            "SAFE": self.hd_space.generate_atomic_vector("SEMANTIC_SAFE"),
            "WARNING": self.hd_space.generate_atomic_vector("SEMANTIC_WARNING"),
            "MARGIN_CALL": self.hd_space.generate_atomic_vector("SEMANTIC_MARGIN_CALL")
        }

    def evaluate_margin_safety(self, current_balance: float) -> tuple[float, str]:
        """
        Evaluates the epistemic safety (Contextual Prior) of the account and 
        collapses the continuous state into a discrete semantic measurement.
        """
        # Downscale balance by 10000.0 to keep phase rotation within bounds.
        # Otherwise an $8000.0 scalar wraps radians hundreds of times aggressively
        scaled_current_bal = current_balance / 10000.0
        v_current_bal = self.hd_space.fractionally_encode(self.scalar_base, scaled_current_bal)
        current_bal_state = self.hd_space.bind(self.r_account, v_current_bal)
        
        # 1. Stratified Prior Check (Contextual EFE)
        # We also need to encode the prior to the exact same scale
        scaled_prior = self.maintenance_margin / 10000.0
        v_prior_margin = self.hd_space.fractionally_encode(self.scalar_base, scaled_prior)
        prior_margin_state = self.hd_space.bind(self.r_account, v_prior_margin)
        
        contextual_efe = self.parietal.assess_contextual_prior(current_bal_state, prior_margin_state)
        
        # 2. Measurement Operator (Collapse to Discrete)
        # Because we only want to penalize dropping *below* the margin, not rising above it,
        # we check the actual phase direction. 
        # A positive balance differential should not trigger a margin call.
        if current_balance >= self.maintenance_margin:
            state_narrative = self.semantic_vocab["SAFE"]
            contextual_efe = 0.0 # Suppress false "danger" for being wealthy
        else:
            if contextual_efe > 0.05:
                # High Free Energy -> Danger State
                state_narrative = self.semantic_vocab["MARGIN_CALL"]
            elif contextual_efe > 0.01:
                state_narrative = self.semantic_vocab["WARNING"]
            else:
                state_narrative = self.semantic_vocab["SAFE"]
            
        # The Parietal Cortex 'snaps' this back to a readable string token
        discrete_token = self.parietal.measurement_operator(state_narrative, self.semantic_vocab)
        
        return contextual_efe, discrete_token

if __name__ == "__main__":
    print("Testing FHRR Stage 6A Quantitative Finance Evaluator...")
    assessor = MarginCallAssessor(10000)
    
    print(f"\n[PRIOR ESTABLISHED] Maintenance Margin requirement: ${assessor.maintenance_margin}")
    print("-" * 70)
    
    # Simulate a market crash drawing down the account
    balances = [8000.0, 6500.0, 5500.0, 5000.0, 4800.0, 4000.0, 2000.0]
    
    for bal in balances:
        efe, discrete_token = assessor.evaluate_margin_safety(bal)
        print(f"Account Balance: ${bal:<7} | Contextual EFE: {efe:.4f}  ->  Measurement: [{discrete_token}]")
