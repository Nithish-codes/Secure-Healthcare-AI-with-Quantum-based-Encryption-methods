import glob, json, os
import numpy as np

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def row(name, r):
    if not r:
        return f"| {name} | n/a | n/a |"
    return f"| {name} | {r['accuracy']:.3f} | {r['auc']:.3f} |"

base, dp, fed = load("results/baseline.json"), load("results/dp.json"), load("results/federated.json")
sec, mem, pipe = load("results/federated_secure.json"), load("results/membership.json"), load("results/pipeline.json")

sweep = {}
for f in sorted(glob.glob("results/sweep/nm*_seed*.json")):
    r = json.load(open(f))
    sweep.setdefault(r["noise_multiplier"], []).append(r)

out = ["# Results", "",
       "Test set: 60 held-out patients (one patient = 1.7 accuracy points).", "",
       "## Single runs with a fixed seed (reference only)", "",
       "| Configuration | Accuracy | AUC |", "|---|---|---|",
       row("Centralized baseline (no privacy)", base),
       row("Centralized DP-SGD (noise 1.0, clip 1.0)", dp and dp.get("1.0")),
       row("Federated, 3 hospitals (noise 0.0)", fed and fed.get("0.0")),
       row("Federated + DP-SGD (noise 1.0)", fed and fed.get("1.0")),
       row("Federated + DP-SGD + secret sharing (noise 1.0)", sec and sec.get("1.0")), ""]

if sweep:
    n = len(next(iter(sweep.values())))
    out += [f"## Full secure pipeline: DP + secret sharing + ML-KEM-768 + AES-256-GCM ({n} seeds per setting)", "",
            "| Noise multiplier | Epsilon (delta 1e-5) | Accuracy (mean +/- std) | AUC (mean +/- std) |",
            "|---|---|---|---|"]
    for nm in sorted(sweep):
        a = np.array([r["accuracy"] for r in sweep[nm]])
        u = np.array([r["auc"] for r in sweep[nm]])
        out.append(f"| {nm} | {sweep[nm][0]['epsilon']} | {a.mean():.3f} +/- {a.std(ddof=1):.3f} "
                   f"| {u.mean():.3f} +/- {u.std(ddof=1):.3f} |")
    worst = max(r["max_diff_vs_plain_average"] for v in sweep.values() for r in v)
    out += ["", "Seeds vary training randomness only; the same test set is used throughout.",
            f"Secret-shared average vs plain average, largest difference over all runs: {worst:.2e}"]
if pipe:
    out.append(f"Transport per run: {pipe['bytes_transmitted']:,} bytes over {pipe['rounds']} rounds, {pipe['seconds']} s")
out.append("")

if mem:
    out += ["## Membership-inference attack (50 trials, 40 training patients)", "",
            "Harsher setup than the pipeline (centralized, 40 patients, batch size 8), so the",
            "accuracy figures here are not comparable to the tables above.", "",
            "| Model | Attack AUC (0.5 = no leakage) | Std. error | Accuracy, unseen patients |",
            "|---|---|---|---|"]
    for k, v in mem.items():
        out.append(f"| {k} | {v['attack_auc']:.3f} | {v['std_error']:.3f} | {v['nonmember_accuracy']:.3f} |")
    out.append("")

open("results/RESULTS.md", "w").write("\n".join(out))
print("\n".join(out))
