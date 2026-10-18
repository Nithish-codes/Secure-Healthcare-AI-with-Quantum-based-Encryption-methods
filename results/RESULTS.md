# Results

Test set: 60 held-out patients (one patient = 1.7 accuracy points).

## Single runs with a fixed seed (reference only)

| Configuration | Accuracy | AUC |
|---|---|---|
| Centralized baseline (no privacy) | 0.833 | 0.950 |
| Centralized DP-SGD (noise 1.0, clip 1.0) | 0.850 | 0.958 |
| Federated, 3 hospitals (noise 0.0) | 0.867 | 0.954 |
| Federated + DP-SGD (noise 1.0) | 0.867 | 0.956 |
| Federated + DP-SGD + secret sharing (noise 1.0) | 0.867 | 0.956 |

## Full secure pipeline: DP + secret sharing + ML-KEM-768 + AES-256-GCM (20 seeds per setting)

| Noise multiplier | Epsilon (delta 1e-5) | Accuracy (mean +/- std) | AUC (mean +/- std) |
|---|---|---|---|
| 1.0 | 41.32 | 0.874 +/- 0.010 | 0.955 +/- 0.002 |
| 2.0 | 14.53 | 0.872 +/- 0.011 | 0.954 +/- 0.002 |
| 5.0 | 4.8 | 0.869 +/- 0.017 | 0.952 +/- 0.006 |
| 10.0 | 2.26 | 0.862 +/- 0.023 | 0.947 +/- 0.009 |

Seeds vary training randomness only; the same test set is used throughout.
Secret-shared average vs plain average, largest difference over all runs: 1.40e-06
Transport per run: 221,040 bytes over 20 rounds, 1.8 s

## Membership-inference attack (50 trials, 40 training patients)

Harsher setup than the pipeline (centralized, 40 patients, batch size 8), so the
accuracy figures here are not comparable to the tables above.

| Model | Attack AUC (0.5 = no leakage) | Std. error | Accuracy, unseen patients |
|---|---|---|---|
| non-private | 0.578 | 0.011 | 0.719 |
| dp noise 0.0 | 0.575 | 0.009 | 0.798 |
| dp noise 1.0 | 0.572 | 0.009 | 0.799 |
| dp noise 2.0 | 0.569 | 0.009 | 0.782 |
| dp noise 5.0 | 0.571 | 0.009 | 0.713 |
| dp noise 10.0 | 0.548 | 0.009 | 0.686 |
