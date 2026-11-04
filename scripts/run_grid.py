"""Reproducible benchmark launcher. Windows: --no-mpi; Linux/WSL or MS-MPI: full grid.

All simulation timings come from the underlying programs, not subprocess overhead.
Each experiment is run as a fresh process, with an explicit run-id and same seed.
"""
import argparse
import csv
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run(cmd, env, timeout):
    print("\n> " + " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.returncode:
        raise RuntimeError(f"command exited with status {proc.returncode}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[100,500,1000,5000,10000])
    p.add_argument("--p", type=int, nargs="+", default=[1,2,4,8])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--no-mpi", action="store_true", help="Only sequential runs; no MPI installed yet")
    p.add_argument("--launcher", default=None, help="Path to mpirun/mpiexec; auto-detect by default")
    p.add_argument("--hwthreads", action="store_true", help="Use Open MPI --use-hwthread-cpus (Linux only)")
    p.add_argument("--tag", default=None, help="Unique experiment prefix for selecting these results")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    if (any(n < 2 for n in args.n) or any(num < 1 for num in args.p)
        or args.steps < 1 or args.repeats < 1 or args.eps <= 0 or args.dt <= 0):
        p.error("require N>=2, P>=1, steps/repeats>=1, dt/eps>0")
    launcher = args.launcher or shutil.which("mpirun") or shutil.which("mpiexec")
    if not args.no_mpi and launcher is None:
        p.error("MPI launcher not found. Run --no-mpi for Windows sequential tests now, "
                "or install an MPI runtime before running the distributed grid")
    tag = args.tag or datetime.now().strftime("grid_%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1", "NUMBA_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
                "MKL_NUM_THREADS":"1", "NBODY_KERNEL":"numba"})
    env.setdefault("HW_TAG", socket.gethostname())
    RESULTS.mkdir(exist_ok=True)
    print(f"Experiment tag: {tag}; hardware tag: {env['HW_TAG']}; MPI={not args.no_mpi}")
    print("Reminder: all runs in a report must be conducted on the same hardware.")
    for n in args.n:
        for repeat in range(1, args.repeats+1):
            seq_id = f"{tag}_seq_n{n}_rep{repeat}"
            base = ["--n",str(n),"--steps",str(args.steps),"--seed",str(args.seed),
                    "--dt",str(args.dt),"--eps",str(args.eps),"--save-every","0"]
            run([sys.executable,"src/sequential.py",*base,"--run-id",seq_id], env, args.timeout)
            if args.no_mpi:
                continue
            for processes in args.p:
                if processes > n:
                    print(f"SKIP P={processes} > N={n}")
                    continue
                mpi_id = f"{tag}_mpi_n{n}_p{processes}_rep{repeat}"
                prefix = [launcher]
                if args.hwthreads and os.name != "nt":
                    prefix.append("--use-hwthread-cpus")
                prefix += ["-np",str(processes)]
                run([*prefix, sys.executable, "src/mpi_nbody.py", *base,"--run-id",mpi_id],
                    env, args.timeout)
                run([sys.executable,"src/compare.py",f"results/final_{seq_id}.npz",
                     f"results/final_{mpi_id}.npz","--tol","1e-9"],env,60)
    print(f"\nCompleted. Run: python scripts/make_plots.py --prefix {tag}")

if __name__ == "__main__":
    main()
