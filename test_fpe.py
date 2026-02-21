import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dhai4_hdc.utils.fpe_math import get_fhrr_space

print("Testing FHRR and FPE Calculus...")
space = get_fhrr_space(10000)

base_price = space.generate_atomic_vector("BASE_PRICE")

# Encode continuous prices: 100, 105, 150
price_100 = space.fractionally_encode(base_price, 100.0)
price_105 = space.fractionally_encode(base_price, 105.0)
price_150 = space.fractionally_encode(base_price, 150.0)

print(f"Price 100 vs Price 100: {space.similarity(price_100, price_100):.4f}")
print(f"Price 100 vs Price 105: {space.similarity(price_100, price_105):.4f}")
print(f"Price 100 vs Price 150: {space.similarity(price_100, price_150):.4f}")

# Geometric Calculus: Delta extraction
# If we extract the delta between 105 and 100, it should geometrically match encoding "5"
delta_5 = space.extract_delta(price_105, price_100)
encoded_5 = space.fractionally_encode(base_price, 5.0)

print(f"Delta(105-100) vs Encoded(5): {space.similarity(delta_5, encoded_5):.4f}")
