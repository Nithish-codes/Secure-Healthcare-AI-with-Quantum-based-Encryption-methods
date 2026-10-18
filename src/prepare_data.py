import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED, N_HOSPITALS, SOURCE = 42, 3, "heart"   # use "breast" for the fallback

if SOURCE == "heart":
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    X = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0] > 0).astype(int)   # 0 = healthy, 1 = disease
    keep = ~X.isna().any(axis=1)
    X, y = X[keep].to_numpy(float), y[keep].to_numpy()
else:
    X, y = load_breast_cancer(return_X_y=True)

# Global held-out test set (never used for training)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Simplification: scaler fit on pooled training data
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# Split training data across simulated hospitals
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X_tr))
for i, part in enumerate(np.array_split(idx, N_HOSPITALS)):
    np.savez(f"data/processed/hospital_{i}.npz",
             X=X_tr[part].astype("float32"), y=y_tr[part].astype("int64"))
    print(f"hospital_{i}: {len(part)} patients, {y_tr[part].mean():.2f} positive rate")

np.savez("data/processed/test.npz",
         X=X_te.astype("float32"), y=y_te.astype("int64"))
print("test:", len(X_te), "patients, features:", X_tr.shape[1])
