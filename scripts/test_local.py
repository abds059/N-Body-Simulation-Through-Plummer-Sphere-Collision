"""Verify physics, decomposition and virtual-rank correctness without any MPI runtime."""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"src"))
from decomposition import decompose
from integrators import drift,kick
from physics import accel
from plummer import two_clusters
from simulated_mpi import simulate


def baseline(n,steps,dt=.01,eps=.05,seed=42):
    pos,vel,mass,labels=two_clusters(n,seed)
    acc=accel(pos,mass,0,n,eps)
    for _ in range(steps):
        kick(vel,acc,.5*dt)
        drift(pos,vel,dt)
        acc=accel(pos,mass,0,n,eps)
        kick(vel,acc,.5*dt)
    return pos,vel


def main():
    # Make sure the numerical checks are never mistaken for performance measurements.
    assert decompose(10,3)==([4,3,3],[0,4,7])
    a=accel(np.array([[0.,0.,0.],[1.,0.,0.]]), np.array([.5,.5]),0,2,1e-6)
    np.testing.assert_allclose(a,[[.5,0,0],[-.5,0,0]],atol=1e-9)
    print("PASS: decomposition and two-body force")
    num_checks=0
    for n in (10,20,50,101):
        seq_pos,seq_vel=baseline(n,steps=20)
        for ranks in (1,2,3,4,5,7,8):
            if ranks>n: continue
            pos,vel,_,_,_=simulate(n=n,ranks=ranks,steps=20)
            dpos=float(np.max(np.abs(pos-seq_pos)))
            dvel=float(np.max(np.abs(vel-seq_vel)))
            print(f"N={n:3d} virtual_P={ranks} |dpos|={dpos:.2e} |dvel|={dvel:.2e} "
                  f"{'PASS' if max(dpos,dvel)<=1e-9 else 'FAIL'}")
            if max(dpos,dvel)>1e-9:
                raise AssertionError(f"Virtual rank results differ at N={n}, P={ranks}")
            num_checks+=1
    print(f"\nALL LOCAL TESTS PASS ({num_checks} virtual-rank comparisons, tol=1e-9)")
    print("These checks do not test real MPI communication or parallel speedup.")

if __name__ == "__main__":
    main()
