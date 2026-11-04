"""Portable MPI end-to-end comparison runner (for Windows MS-MPI or Linux Open MPI)."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent


def call(cmd,env):
    print("\n> "+ " ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launcher",default=None)
    ap.add_argument("--hwthreads",action="store_true")
    ap.add_argument("--quick",action="store_true",help="Small smoke tests instead of full grid")
    args=ap.parse_args()
    launcher=args.launcher or shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        ap.error("MPI not installed; run 'python scripts/test_local.py' first")
    env=os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
                "NBODY_KERNEL":"numba"})
    cases=[(20,[2,3,4,8]),(101,[3,5,7,8])] if args.quick else [
        (10,[8]),(20,[8]),(50,[8]),(1000,[5,7,8]),(1001,[5,7,8])]
    for n,ps in cases:
        sid=f"verify_seq_{n}"
        call([sys.executable,"src/sequential.py","--n",str(n),"--steps","100","--run-id",sid],env)
        for p in ps:
            mid=f"verify_mpi_{n}_p{p}"
            cmd=[launcher]
            if args.hwthreads and os.name!="nt":cmd.append("--use-hwthread-cpus")
            cmd += ["-np",str(p),sys.executable,"src/mpi_nbody.py","--n",str(n),
                    "--steps","100","--run-id",mid]
            call(cmd,env)
            call([sys.executable,"src/compare.py",f"results/final_{sid}.npz",
                  f"results/final_{mid}.npz","--tol","1e-9"],env)
    print("ALL REAL MPI COMPARISONS PASS")

if __name__=="__main__":main()
