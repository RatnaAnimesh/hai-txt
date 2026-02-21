import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dhai4_hdc.models.level3_parietal import HDC_ParietalCortex

print("Testing Level 3 Parietal Cortex (Physical Grounding via FHRR)...")
parietal = HDC_ParietalCortex(dimension=10000)

# State 1: A 10kg mass moving at 10 m/s
m1 = 10.0
v1 = 10.0
print(f"State 1: Mass={m1}kg, Vel={v1}m/s.  (Actual KE = 500J)")

# State 2 (Valid Transition): The mass splits but accelerates flawlessly conserving energy
# Mass = 20kg, Vel = sqrt(50) = 7.071 m/s (KE = 0.5 * 20 * 50 = 500J)
m2 = 20.0
v2 = np.sqrt(50)
print(f"State 2 (Physically Valid Proposed Future): Mass={m2}kg, Vel={v2:.3f}m/s.  (Actual KE = 500J)")

# State 3 (Invalid Transition): Magic acceleration that creates energy!
m3 = 10.0
v3 = 20.0
print(f"State 3 (Physically Invalid Proposed Future): Mass={m3}kg, Vel={v3}m/s.  (Actual KE = 2000J)")

print("-" * 50)
print("Evaluating Active Inference Prior Divergence (Expected Free Energy):")

efe_valid = parietal.assess_transition_physics(m1, v1, m2, v2)
print(f"EFE of transition 1 -> 2: {efe_valid:.4f}")
if efe_valid < 0.01:
    print("  [PASS] Parietal Cortex correctly evaluated the geometric Prior and permitted the physically sound transition.")
else:
    print("  [FAIL] Parietal Cortex rejected a physically valid move.")

print("\n")

efe_invalid = parietal.assess_transition_physics(m1, v1, m3, v3)
print(f"EFE of transition 1 -> 3: {efe_invalid:.4f}")
if efe_invalid > 0.5:
    print("  [PASS] Parietal Cortex caught the Conservation of Energy violation purely via massive FHRR geometry divergence! Immediate Epistemic Pruning triggered.")
else:
    print("  [FAIL] Parietal Cortex failed to recognize the physical impossibility.")
