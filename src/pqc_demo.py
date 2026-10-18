import os
import numpy as np
from kyber_py.ml_kem import ML_KEM_768
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Server publishes an ML-KEM public key, keeps the private key
ek, dk = ML_KEM_768.keygen()

def session_key(shared):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"fl-update-v1").derive(shared)

# Hospital: encapsulate -> shared secret + KEM ciphertext, then encrypt the update
update = np.random.default_rng(0).normal(size=14).astype(np.float64)  # stand-in for a model update
shared_h, kem_ct = ML_KEM_768.encaps(ek)
nonce = os.urandom(12)
sealed = AESGCM(session_key(shared_h)).encrypt(nonce, update.tobytes(), b"hospital_0|round_1")

# Server: decapsulate with the private key, then decrypt
shared_s = ML_KEM_768.decaps(dk, kem_ct)
plain = AESGCM(session_key(shared_s)).decrypt(nonce, sealed, b"hospital_0|round_1")
recovered = np.frombuffer(plain, dtype=np.float64)

print("shared secrets match:", shared_h == shared_s)
print("update recovered exactly:", np.array_equal(update, recovered))
print("KEM ciphertext bytes:", len(kem_ct), "| sealed update bytes:", len(sealed))
