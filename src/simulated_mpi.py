"""Emulate MPI's block decomposition in one Python process, WITHOUT using MPI.

Only for algorithm/correctness testing on Windows before installing MPI.
DO NOT use timings from this script for distributed speedup claims.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from decomposition import decompose
from integrators import drift, kick
from physics import accel
from plummer import two_clusters
from sequential import RESULTS


def simulate(n=100, ranks=4, steps=100, dt=0.01, eps=0.05, seed=42,
             sep=8.0, vfrac=0.6, impact=1.0, save_every=0):
    """Return final state and optional saved positions, emulating one Allgatherv/step."""
    if not (2 <= n and 1 <= ranks <= n and steps >= 1 and dt > 0 and eps > 0 and save_every >= 0):
        raise ValueError("require n>=2, 1<=ranks<=n, steps>=1, dt>0, eps>0 and save_every>=0")
    pos, vel, mass, labels = two_clusters(n, seed, sep, vfrac, impact)
    counts, displs = decompose(n, ranks)
    local_vel = [vel[s:s+c].copy() for s, c in zip(displs, counts)]
    local_acc = [accel(pos, mass, s, s+c, eps) for s, c in zip(displs, counts)]
    next_pos = np.empty_like(pos)
    frames = [pos.copy()] if save_every else []

    for step in range(1, steps+1):
        for s, c, v, a in zip(displs, counts, local_vel, local_acc):
            kick(v, a, 0.5*dt)
            next_pos[s:s+c] = pos[s:s+c]
            drift(next_pos[s:s+c], v, dt)
        # Equivalent to all ranks receiving the assembled global positions.
        pos, next_pos = next_pos, pos
        for i, (s, c) in enumerate(zip(displs, counts)):
            local_acc[i] = accel(pos, mass, s, s+c, eps)
            kick(local_vel[i], local_acc[i], 0.5*dt)
        if save_every and step % save_every == 0:
            frames.append(pos.copy())
    result_vel = np.concatenate(local_vel, axis=0)
    return pos, result_vel, mass, labels, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep", type=float, default=8.0)
    ap.add_argument("--vfrac", type=float, default=0.6)
    ap.add_argument("--impact", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    try:
        pos, vel, mass, label, frames = simulate(args.n, args.ranks, args.steps,
            args.dt, args.eps, args.seed, args.sep, args.vfrac, args.impact, args.save_every)
    except ValueError as exc:
        ap.error(str(exc))
    run_id = args.run_id or f"virtual_N{args.n}_P{args.ranks}_{int(time.time())}"
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"final_{run_id}.npz"
    np.savez(path, pos=pos, vel=vel, mass=mass, label=label)
    if frames:
        np.savez(RESULTS / f"frames_{run_id}.npz", frames=np.array(frames))
    print(f"VIRTUAL MPI ONLY (not actual parallel execution): N={args.n} P={args.ranks} "
          f"steps={args.steps} output={path}")

if __name__ == "__main__":
    main()
