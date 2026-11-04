"""Compare optional ring-exchange variant against sequential on an installed MPI runtime."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent


def call(command,env):
    print("\n> "+" ".join(map(str,command)),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--launcher",default=None)
    a.add_argument("--hwthreads",action="store_true")
    a.add_argument("--quick",action="store_true")
    args=a.parse_args()
    launcher=args.launcher or shutil.which("mpirun") or shutil.which("mpiexec")
    if not launcher:a.error("MPI runtime not available; local correctness tests can run without it")
    env=os.environ.copy();env.update({"OMP_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1",
                "NBODY_KERNEL":"numba","OPENBLAS_NUM_THREADS":"1"})
    for n,ps in ([(30,(1,2,3))] if args.quick else [(100,(1,2,3,4)),(1001,(3,5,7))]):
        sid=f"ring_validation_seq_{n}"
        call([sys.executable,"src/sequential.py","--n",str(n),"--steps","100","--run-id",sid],env)
        for p in ps:
            rid=f"ring_validation_n{n}_p{p}"
            prefix=[launcher]
            if args.hwthreads and os.name!="nt":prefix.append("--use-hwthread-cpus")
            call([*prefix,"-np",str(p),sys.executable,"src/mpi_ring.py", "--n",str(n),
                  "--steps","100","--run-id",rid],env)
            call([sys.executable,"src/compare.py",f"results/final_{sid}.npz",
                  f"results/final_{rid}.npz","--tol","1e-9"],env)
    print("RING MPI COMPARISONS PASS")

if __name__=="__main__":main()
