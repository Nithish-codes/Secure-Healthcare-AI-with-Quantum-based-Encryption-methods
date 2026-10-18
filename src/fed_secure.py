import os, json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

rng = np.random.default_rng(42)          # used only for training randomness
SCALE = 2**20                            # fixed-point precision
N_NODES = 3                              # validation nodes holding shares

parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
H = [(p["X"].astype(np.float64), p["y"].astype(np.float64)) for p in parts]
test = np.load("data/processed/test.npz")
Xt, yt = test["X"].astype(np.float64), test["y"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def local_train(w, b, X, y, epochs, lr=0.1, clip=1.0, nm=0.0, batch=32):
    w = w.copy()
    n = len(X)
    bs = min(batch, n)
    for _ in range(epochs):
        for _ in range(max(1, n // bs)):
            idx = rng.choice(n, bs, replace=False)
            err = sigmoid(X[idx] @ w + b) - y[idx]
            g = np.hstack([X[idx] * err[:, None], err[:, None]])
            g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / clip)
            step = (g.sum(0) + rng.normal(0, nm * clip, g.shape[1])) / bs
            w -= lr * step[:-1]
            b -= lr * step[-1]
    return w, b

# ---- secret sharing (additive, mod 2^64, cryptographically secure randomness) ----
def encode(v):
    return np.round(v * SCALE).astype(np.int64).view(np.uint64)

def decode(u):
    return u.view(np.int64).astype(np.float64) / SCALE

def secure_random(n):
    return np.frombuffer(os.urandom(8 * n), dtype=np.uint64).copy()

def share(vec):
    """Split an encoded vector into N_NODES shares that sum to it (mod 2^64)."""
    shares = [secure_random(len(vec)) for _ in range(N_NODES - 1)]
    last = vec.copy()
    for s in shares:
        last = last - s
    return shares + [last]

def secure_average(updates, weights):
    """updates: list of vectors (one per hospital); weights sum to 1."""
    node_sums = [np.zeros(len(updates[0]), dtype=np.uint64) for _ in range(N_NODES)]
    for u, wt in zip(updates, weights):
        for node, s in enumerate(share(encode(u * wt))):
            node_sums[node] = node_sums[node] + s        # each node sees only random shares
    total = np.zeros_like(node_sums[0])
    for s in node_sums:                                   # server combines node sums only
        total = total + s
    return decode(total)

def federated_secure(nm, rounds=20, local_epochs=3):
    d = H[0][0].shape[1]
    w, b = np.zeros(d), 0.0
    sizes = np.array([len(x) for x, _ in H], float)
    wts = sizes / sizes.sum()
    worst = 0.0
    for _ in range(rounds):
        ups = [local_train(w, b, x, y, local_epochs, nm=nm) for x, y in H]
        vecs = [np.append(u[0], u[1]) for u in ups]
        agg = secure_average(vecs, wts)
        plain = sum(wt * v for wt, v in zip(wts, vecs))
        worst = max(worst, float(np.abs(agg - plain).max()))
        w, b = agg[:-1], agg[-1]
    return w, b, worst

results = {}
for nm in [0.0, 1.0]:
    w, b, diff = federated_secure(nm)
    p = sigmoid(Xt @ w + b)
    results[str(nm)] = {"accuracy": float(accuracy_score(yt, p > 0.5)),
                        "auc": float(roc_auc_score(yt, p)),
                        "max_diff_vs_plain_average": diff}
    print(f"secure federated, noise multiplier {nm}: {results[str(nm)]}")

json.dump(results, open("results/federated_secure.json", "w"), indent=2)
