import time
import numpy as np
from phe import paillier

rng = np.random.default_rng(0)
t0 = time.time()
pub, priv = paillier.generate_paillier_keypair(n_length=2048)   # held by a key authority, not the server
print(f"keygen: {time.time() - t0:.1f} s")

D = 15                                              # 14 weights + bias, as in the pipeline
updates = [rng.normal(size=D) for _ in range(3)]    # stand-in hospital updates
sizes = np.array([100.0, 100.0, 97.0])
wts = sizes / sizes.sum()

# Each hospital encrypts its update with the public key
t0 = time.time()
enc = [[pub.encrypt(float(v)) for v in u] for u in updates]
print(f"encrypt 3 updates: {time.time() - t0:.1f} s")

# Server: weighted sum computed on ciphertexts only, no private key
t0 = time.time()
agg = [sum(enc[h][j] * float(wts[h]) for h in range(3)) for j in range(D)]
print(f"homomorphic aggregation: {time.time() - t0:.2f} s")

# Key authority decrypts only the aggregate
result = np.array([priv.decrypt(c) for c in agg])
plain = sum(w * u for w, u in zip(wts, updates))
print("max difference vs plain average:", float(np.abs(result - plain).max()))
