"""Fit an exploratory MPI per-step cost model to repeated runs.csv measurements.

Model: Tstep = k * N^2/P + a * log2(P) + b * 24*N, with k,a,b >= 0.
The last term is a fitted N-dependent cost, NOT a literal network transfer at P=1.
No claim about a scaling crossover should be made without measured confirmation.
"""
import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def nonnegative_fit(X, y):
    """Small 3-column nonnegative least squares without a SciPy dependency."""
    scale = np.maximum(np.linalg.norm(X, axis=0), 1e-30)
    A = X / scale
    best = None
    best_error = float("inf")
    for n_active in range(4):
        for active in combinations(range(3), n_active):
            coef = np.zeros(3)
            if active:
                partial, *_ = np.linalg.lstsq(A[:, active], y, rcond=None)
                if np.any(partial < 0):
                    continue
                coef[list(active)] = partial
            error = float(np.sum((A @ coef - y) ** 2))
            if error < best_error:
                best_error, best = error, coef
    return best / scale


def features(N, P):
    return [N * N / P, math.log2(P), 24 * N]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=RESULTS / "runs.csv")
    ap.add_argument("--hardware-tag", required=True, help="Use only one benchmark machine")
    ap.add_argument("--run-prefix", default="", help="Only these run IDs, e.g. bench_")
    ap.add_argument("--min-n", type=int, default=500, help="Ignore tiny timing-noise cases")
    ap.add_argument("--steps", type=int, default=100, help="Require equal step counts across runs")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict-n", type=int, default=5000)
    ap.add_argument("--max-p", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS / "costmodel.png")
    args = ap.parse_args()
    if args.predict_n < 2 or args.min_n < 2 or args.max_p < 2:
        ap.error("need predict-n >= 2, min-n >= 2 and max-p >= 2")

    groups = defaultdict(list)
    if not args.csv.is_file():
        ap.error(f"missing {args.csv}")
    with args.csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                if (row["impl"] != "mpi" or row["hardware_tag"] != args.hardware_tag
                    or not row["run_id"].startswith(args.run_prefix)):
                    continue
                N, P = int(row["N"]), int(row["P"])
                steps, t_total = int(row["steps"]), float(row["t_total"])
                if (N < args.min_n or P < 2 or steps != args.steps or t_total <= 0
                    or float(row["dt"]) != args.dt or float(row["eps"]) != args.eps
                    or int(row["seed"]) != args.seed):
                    continue
                groups[(N, P)].append(t_total / steps)
            except (KeyError, ValueError, TypeError):
                continue

    if len(groups) < 4 or len({n for n, p in groups}) < 2 or len({p for n, p in groups}) < 2:
        ap.error("need >=4 (N,P) cases, >=2 body counts and >=2 P values on one hardware tag; "
                 "run the benchmark grid first")
    items = sorted((n, p, float(np.median(ts)), len(ts))
                   for (n, p), ts in groups.items())
    X = np.array([features(n, p) for n, p, t, r in items], dtype=float)
    y = np.array([t for n, p, t, r in items])
    k, a, b = nonnegative_fit(X, y)
    predictions = X @ np.array([k, a, b])
    relative_error = np.abs(predictions - y) / y
    print("FIT (MPI P>=2; same machine; median time/step per N,P)")
    print(f"cases={len(items)} repeats/case=" + ", ".join(str(r) for _, _, _, r in items))
    print(f"k={k:.9e} s/body-pair, a={a:.9e} s, b={b:.9e} s/byte-equivalent")
    print(f"median relative fit error={np.median(relative_error):.1%}")
    if any(repeats < 3 for _, _, _, repeats in items):
        print("WARNING: fewer than three repeats in some cases; fit is preliminary.")
    if np.median(relative_error) > 0.25:
        print("WARNING: fit error >25%; model cannot support a credible crossover claim.")
    print("     N    P   repeats   observed(s/step)  predicted(s/step)   error")
    for (n, p, observed, repeats), predicted in zip(items, predictions):
        print(f"{n:6d} {p:4d} {repeats:9d} {observed:18.6f} {predicted:18.6f} "
              f"{abs(predicted-observed)/observed:7.1%}")

    Ps = np.arange(2, args.max_p + 1)
    predicted = np.array([np.dot(features(args.predict_n, int(p)), [k, a, b]) for p in Ps])
    best_p = int(Ps[np.argmin(predicted)])
    print(f"At N={args.predict_n}, fitted minimum among P=2..{args.max_p}: P={best_p}")
    if best_p == args.max_p:
        print("No predicted crossover within this range; do not extrapolate beyond observed hardware.")
    print("Model is exploratory: 24*N is logical payload, not actual network traffic; "
          "P=1 was excluded because no inter-rank communication occurs there.")

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(Ps, predicted, "o-", label="Model prediction")
    same_n = [(p, t) for n, p, t, r in items if n == args.predict_n]
    if same_n:
        ax.scatter([p for p, t in same_n], [t for p, t in same_n],
                   s=70, label="Measured MPI median")
    ax.set_xticks(Ps)
    ax.set_xlabel("MPI processes (P)")
    ax.set_ylabel("Time per step (seconds)")
    ax.set_title(f"Exploratory cost model | N={args.predict_n}")
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"chart: {args.out}")


if __name__ == "__main__":
    main()
