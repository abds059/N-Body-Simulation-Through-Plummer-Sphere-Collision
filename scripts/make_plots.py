"""Produce performance CSV and graphs from measured results/runs.csv (no fabricated data)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def summarize(input_path, prefix="", hardware_tag=""):
    groups = defaultdict(list)
    with input_path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                if prefix and not r["run_id"].startswith(prefix):
                    continue
                if hardware_tag and r["hardware_tag"] != hardware_tag:
                    continue
                n, p = int(r["N"]), int(r["P"])
                times = [float(r[k]) for k in ("t_total","t_comp","t_comm")]
                if min(times) < 0 or times[0] <= 0:
                    continue
                groups[(r["impl"], n, p, int(r["steps"]), float(r["dt"]),
                        float(r["eps"]), int(r["seed"]),r["hardware_tag"])].append(times)
            except (KeyError,ValueError,TypeError):
                continue
    summary = []
    for key, values in sorted(groups.items()):
        impl,n,p,steps,dt,eps,seed,hardware = key
        a = np.array(values)
        summary.append(dict(impl=impl,N=n,P=p,steps=steps,dt=dt,eps=eps,seed=seed,
                            hardware_tag=hardware,repeats=len(values),
                            t_total_median=float(np.median(a[:,0])),
                            t_total_std=float(np.std(a[:,0],ddof=1)) if len(a)>1 else 0.0,
                            t_comp_median=float(np.median(a[:,1])),
                            t_comm_median=float(np.median(a[:,2]))))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=RESULTS / "runs.csv")
    ap.add_argument("--prefix", default="", help="Filter a specific benchmark tag")
    ap.add_argument("--hardware-tag", default="")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "plots")
    args = ap.parse_args()
    if not args.csv.is_file():
        ap.error(f"Missing data: {args.csv}. Run simulations first.")
    rows = summarize(args.csv, args.prefix, args.hardware_tag)
    if not rows:
        ap.error("No compatible measurements found; check prefix and hardware tag")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq = {(r["N"],r["steps"],r["dt"],r["eps"],r["seed"],r["hardware_tag"]):r
           for r in rows if r["impl"] == "seq"}
    for r in rows:
        r["speedup"] = r["efficiency"] = ""
        if r["impl"] == "mpi":
            s = seq.get((r["N"],r["steps"],r["dt"],r["eps"],r["seed"],r["hardware_tag"]))
            if s:
                r["speedup"] = s["t_total_median"]/r["t_total_median"]
                r["efficiency"] = r["speedup"]/r["P"]
    summary_path = args.out_dir / "performance_summary.csv"
    with summary_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {summary_path} ({len(rows)} grouped conditions)")
    ns = sorted({r["N"] for r in rows})
    seqns = [r for r in rows if r["impl"] == "seq"]
    if seqns:
        fig, ax = plt.subplots(figsize=(7.8,4.8))
        for hardware in sorted({r["hardware_tag"] for r in seqns}):
            subset = sorted((r for r in seqns if r["hardware_tag"] == hardware),key=lambda r:r["N"])
            ax.plot([r["N"] for r in subset], [r["t_total_median"] for r in subset],"o-",label=hardware)
        ax.set(xlabel="Number of bodies (N)",ylabel="Sequential time (s)",title="Sequential baseline")
        ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir/"sequential_time.png",dpi=150); plt.close(fig)
    mpi = [r for r in rows if r["impl"] == "mpi" and r["speedup"] != ""]
    if not mpi:
        print("Only sequential measurements found. MPI speedup plots will appear after MPI runs.")
        return
    for metric, filename, ylabel in (("speedup","speedup.png","Sequential/MPI time"),
                                      ("efficiency","efficiency.png","Speedup / P"),
                                      ("t_total_median","mpi_total_time.png","Median MPI time (s)")):
        fig, ax = plt.subplots(figsize=(8,5))
        for n in ns:
            subset = sorted((r for r in mpi if r["N"] == n),key=lambda r:r["P"])
            if subset:
                ax.plot([r["P"] for r in subset],[r[metric] for r in subset],"o-",label=f"N={n}")
        ax.set(xlabel="Number of MPI ranks (P)",ylabel=ylabel,title=metric.replace("_"," ").title())
        if metric == "speedup":
            ax.axline((0,0),(1,1),linestyle="--",color="gray",label="Ideal speedup")
        if metric == "efficiency":
            ax.axhline(1,linestyle="--",color="gray",label="Ideal efficiency")
        ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir/filename,dpi=150); plt.close(fig)
        print("Saved",args.out_dir/filename)
    fig,ax=plt.subplots(figsize=(8,5))
    p_candidates=sorted({r["P"] for r in mpi})
    for p in p_candidates:
        sample=sorted((r for r in mpi if r["P"] == p),key=lambda r:r["N"])
        if sample:
            ax.plot([r["N"] for r in sample], [r["t_comm_median"]/r["t_total_median"] for r in sample],
                    "o-",label=f"P={p}")
    ax.set(xlabel="Number of bodies (N)",ylabel="Mean-rank communication / wall time",
           title="Communication time fraction (approximate)")
    ax.legend();fig.tight_layout();fig.savefig(args.out_dir/"communication_fraction.png",dpi=150);plt.close(fig)
    print("Saved",args.out_dir/"communication_fraction.png")

if __name__ == "__main__":
    main()
