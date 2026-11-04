"""Show measured and theoretical per-rank balance from a rank_times_<id>.csv file."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", help="The MPI run id (without rank_times_ prefix)")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()

    path = args.results / f"rank_times_{args.run_id}.csv"
    if not path.is_file():
        ap.error(f"not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        ap.error("rank-time file has no rows")
    try:
        rows.sort(key=lambda r: int(r["rank"]))
        ranks = np.array([int(r["rank"]) for r in rows])
        bodies = np.array([int(r["bodies"]) for r in rows])
        comp = np.array([float(r["t_comp"]) for r in rows])
        comm = np.array([float(r["t_comm"]) for r in rows])
    except (KeyError, TypeError, ValueError) as e:
        ap.error(f"invalid rank CSV: {e}")
    P = len(rows)
    N = int(bodies.sum())
    if not np.array_equal(ranks, np.arange(P)) or np.any(bodies < 1):
        ap.error("ranks must be 0..P-1 and each rank must own at least one body")
    if np.any(comp < 0) or np.any(comm < 0) or comp.mean() <= 0:
        ap.error("timings must be nonnegative with a nonzero mean compute time")

    theoretical = bodies.max() / bodies.mean()
    measured = comp.max() / comp.mean()
    expected = comp.mean() * bodies / bodies.mean()

    print(f"run_id={args.run_id} N={N} P={P}")
    print(f"measured compute imbalance max/mean={measured:.4f}")
    print(f"theoretical work imbalance max/mean={theoretical:.4f}")
    print("rank  bodies   t_comp(s)   t_comm(s)   compute/mean  bodies/mean")
    for r in range(P):
        print(f"{r:4d}  {bodies[r]:6d}   {comp[r]:9.6f}   {comm[r]:9.6f}"
              f"   {comp[r]/comp.mean():12.4f}   {bodies[r]/bodies.mean():11.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(P)
    ax.bar(x, comp, label="Measured compute time")
    ax.plot(x, expected, "o--", label="Expected from body counts (mean-normalized)")
    ax.set_xticks(x, [str(r) for r in ranks])
    ax.set_xlabel("MPI rank")
    ax.set_ylabel("Accumulated compute time (seconds)")
    ax.set_title(f"Load balance | N={N}, P={P} | max/mean={measured:.3f}")
    ax.legend()
    fig.tight_layout()
    output = args.results / f"loadbalance_{args.run_id}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"chart: {output}")
    print("Note: measured imbalance also includes OS scheduling and timing noise.")


if __name__ == "__main__":
    main()
