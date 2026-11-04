"""Compare MPI positions-only versus full-state communication, on any MPI-capable OS.

Real MPI required; output saved to regular runs.csv plus an analysis CSV.
"""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/"results"


def run(command,env):
    print("\n> "+" ".join(map(str,command)),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=2000)
    p.add_argument("--p",type=int,default=4)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--launcher",default=None)
    p.add_argument("--hwthreads",action="store_true")
    a=p.parse_args()
    if a.n<a.p or a.p<1 or a.n<2 or a.steps<1 or a.repeats<1:
        p.error("require N >= P >= 1, N>=2, steps>=1, repeats>=1")
    launcher=a.launcher or shutil.which("mpirun") or shutil.which("mpiexec")
    if not launcher:p.error("No MPI runtime found. The local simulator cannot measure communication")
    env=os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
                "MKL_NUM_THREADS":"1","NBODY_KERNEL":"numba"})
    tag=datetime.now().strftime("comm_%Y%m%d_%H%M%S")
    prefix=[launcher]
    if a.hwthreads and os.name!="nt":prefix.append("--use-hwthread-cpus")
    prefix += ["-np",str(a.p),sys.executable,"src/mpi_nbody.py"]
    for i in range(a.repeats):
        sid=f"{tag}_seq_{i}"
        common=["--n",str(a.n),"--steps",str(a.steps)]
        run([sys.executable,"src/sequential.py",*common,"--run-id",sid],env)
        for mode in ("positions","full-state"):
            rid=f"{tag}_{mode}_{i}"
            run([*prefix,*common,"--run-id",rid,"--comm-mode",mode],env)
            run([sys.executable,"src/compare.py",f"results/final_{sid}.npz",
                 f"results/final_{rid}.npz","--tol","1e-9"],env)
    table={"positions":[],"full-state":[]}
    with (RESULTS/"runs.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["run_id"].startswith(tag) and r["impl"]=="mpi":
                mode="full-state" if "_full-state_" in r["run_id"] else "positions"
                table[mode].append((float(r["t_total"]),float(r["t_comm"])))
    for mode in ("positions","full-state"):
        arr=np.array(table[mode],dtype=float)
        if arr.shape!=(a.repeats,2):raise RuntimeError(f"Incomplete data for {mode}")
        payload=a.n*(24 if mode=="positions" else 56)
        print(f"{mode:11s}: {payload:9d} logical bytes per step, "
              f"median total={np.median(arr[:,0]):.6f}s median communication={np.median(arr[:,1]):.6f}s")
    print("The logical payload counts each global state once; it is not total network traffic.")
    print(f"Individual measurement rows are in results/runs.csv (prefix {tag}).")

if __name__=="__main__":main()
