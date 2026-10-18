# Privacy-Preserving Healthcare AI with Quantum-Safe Encryption

Prototype of privacy-preserving diagnostic model training, run on Linux on IBM Z (s390x).
Dataset: UCI Heart Disease (open, de-identified), split across 3 simulated hospitals.

## Pipeline
1. Each hospital trains locally with DP-SGD (per-example clipping + Gaussian noise).
2. Each hospital splits its model update into additive secret shares, one per validation node.
3. Each share is encrypted for its node with ML-KEM-768 key exchange and AES-256-GCM.
4. Nodes decrypt and sum their shares; the server combines node sums into the new global model.
5. The server never sees any single hospital's update.

## Run it
Prerequisites (Ubuntu on s390x; scipy has no s390x wheel, so use the apt packages):
    sudo apt install -y python3-dev python3-numpy python3-scipy python3-pandas python3-sklearn python3-cryptography
    python3 -m venv --system-site-packages .venv && source .venv/bin/activate
    pip install ucimlrepo kyber-py phe

Pipeline (epsilon.py must run before the sweep):
    python src/prepare_data.py
    python src/baseline.py
    python src/dp_train.py
    python src/fed_train.py
    python src/fed_secure.py
    python src/pqc_demo.py
    python src/he_demo.py
    python src/membership_test.py
    python src/epsilon.py
    mkdir -p results/sweep
    for nm in 1.0 2.0 5.0 10.0; do for seed in $(seq 1 20); do NM=$nm SEED=$seed OUT=results/sweep/nm${nm}_seed${seed}.json python src/secure_pipeline.py > /dev/null; done; done
    python src/make_report.py

Results are written to results/ (see results/RESULTS.md).

## Findings
- The full secure pipeline matched the non-private baseline within test-set noise.
- Secret-shared aggregation matched plain averaging to about 1e-6.
- A loss-based membership-inference attack reached AUC about 0.57 on the non-private model.
  DP noise up to 5.0 produced no measurable reduction; at noise 10.0 the attack AUC fell to
  0.548 (about 2 standard errors lower), but unseen-patient accuracy fell from about 0.80
  to 0.69. The measured privacy benefit is small and comes at a clear accuracy cost.

## Limitations (please read)
- Everything runs on one machine as a simulation of 3 hospitals and 3 validation nodes.
- No formal privacy budget (epsilon) was computed.
- Hardware TEE (IBM Secure Execution) was not enabled in this run; the design is TEE-ready.
- Homomorphic aggregation was demonstrated separately (src/he_demo.py) with Paillier (additive HE)
  on stand-in updates: matched the plain average to 1e-16, but is slow in pure Python and is NOT
  post-quantum. Lattice-based HE (CKKS/BFV via TenSEAL) could not be built on s390x in this run.
- kyber-py is a pure-Python reference implementation, not hardened for production.
- Hospital authentication with post-quantum signatures is not implemented.
- The 60-patient test set is small; small accuracy differences are not meaningful.
- Data scaling used pooled statistics, a simplification versus real federated deployment.

## Headline configuration
Noise multiplier 10.0, clip norm 1.0, 20 rounds x 3 local epochs, 3 hospitals of 79 patients each.
Epsilon about 2.26 at delta 1e-5 (per hospital, record-level, Renyi-DP accountant with a
Poisson-sampling approximation, counting every round's update as visible).
Over 20 seeds: accuracy 0.862 +/- 0.023, AUC 0.947 +/- 0.009, versus 0.874 +/- 0.010 and
0.955 +/- 0.002 at noise 1.0 (epsilon about 41.3). Reducing epsilon from 41 to 2.3 cost about
1 accuracy point (less than one test patient) and about 0.008 AUC. The test set has 60 patients,
and the standard deviations reflect training randomness only.
The membership-inference experiment used a harsher setup (centralized, 40 patients, batch 8),
so its accuracy figures are not comparable to the pipeline results.
Full tables: results/RESULTS.md

## Saved artifacts
- data/raw/heart_disease_uci.csv: raw copy of the public UCI dataset (src/save_raw_data.py)
- models/global_model_nm10.npz: one example global model (noise 10.0, seed 42; src/secure_pipeline.py with MODEL_OUT set).
  Single-run metrics vary; the 20-seed table above is the reference result.
- Inference check: python3 src/predict.py
