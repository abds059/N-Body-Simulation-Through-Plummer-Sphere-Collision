"""MPI distributed N-body: default uses one position Allgatherv per step."""
import argparse
import csv
import os
import socket
import sys
import time

import numpy as np
from mpi4py import MPI

from decomposition import decompose
from integrators import drift, kick
from physics import accel, energy
from plummer import two_clusters
from sequential import COLUMNS, RESULTS


def main():
    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep", type=float, default=8.0)
    ap.add_argument("--vfrac", type=float, default=0.6)
    ap.add_argument("--impact", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--comm-mode", choices=("positions", "full-state"), default="positions",
                    help="Experimental: full-state gathers unnecessary velocities and masses too")
    a = ap.parse_args()

    err = None
    if a.n < 2 or a.steps < 1 or a.dt <= 0 or a.eps <= 0 or a.save_every < 0:
        err = "need n >= 2, steps >= 1, dt > 0, eps > 0, save-every >= 0"
    elif a.n < P:
        err = f"need n >= number of processes (n={a.n}, P={P})"
    if err:
        if rank == 0:
            print("error:", err, file=sys.stderr)
        sys.exit(2)
    rid = comm.bcast(a.run_id or f"mpi_N{a.n}_P{P}_s{a.steps}_{int(time.time())}", root=0)

    # every rank builds the same start from the same seed (no broadcast needed)
    pos, vel, mass, label = two_clusters(a.n, a.seed, a.sep, a.vfrac, a.impact)
    counts, displs = decompose(a.n, P)
    s, e = displs[rank], displs[rank] + counts[rank]
    rc = [3 * c for c in counts]  # positions are N x 3 doubles
    rd = [3 * d for d in displs]
    my_vel = vel[s:e].copy()  # velocities never leave their owner
    pos_new = np.empty_like(pos)
    if a.comm_mode == "full-state":
        # Preallocate buffers; all experimental extra communication is inside the timer.
        vel_received = np.empty_like(vel)
        mass_received = np.empty_like(mass)
    save = bool(a.save_every) and rank == 0
    frames = [pos.copy()] if save else []
    if rank == 0:
        k0, w0 = energy(pos, vel, mass, a.eps)
    accel(pos[:2], mass[:2], 0, 2, a.eps)  # warm-up (Numba compile) outside the timer

    t_comp = t_comm = 0.0
    comm.Barrier()  # clean start and stop of the clock only
    t0 = MPI.Wtime()
    c0 = MPI.Wtime()
    my_acc = accel(pos, mass, s, e, a.eps)
    t_comp += MPI.Wtime() - c0
    for step in range(1, a.steps + 1):
        c0 = MPI.Wtime()
        kick(my_vel, my_acc, 0.5 * a.dt)
        local_new = pos[s:e].copy()  # new positions go into a new array
        drift(local_new, my_vel, a.dt)
        c1 = MPI.Wtime()
        comm.Allgatherv(local_new, [pos_new.reshape(-1), rc, rd, MPI.DOUBLE])
        if a.comm_mode == "full-state":
            # These two global arrays are never used by the integrator.
            # Three collectives vs one: this measures both extra bytes AND latency.
            comm.Allgatherv(my_vel, [vel_received.reshape(-1), rc, rd, MPI.DOUBLE])
            comm.Allgatherv(mass[s:e], [mass_received, counts, displs, MPI.DOUBLE])
        c2 = MPI.Wtime()
        pos, pos_new = pos_new, pos
        my_acc = accel(pos, mass, s, e, a.eps)
        kick(my_vel, my_acc, 0.5 * a.dt)
        c3 = MPI.Wtime()
        t_comp += (c1 - c0) + (c3 - c2)
        t_comm += c2 - c1  # includes waiting for the slowest rank
        if save and step % a.save_every == 0:
            frames.append(pos.copy())
    comm.Barrier()
    t_total = MPI.Wtime() - t0

    vel_full = np.empty((a.n, 3)) if rank == 0 else None
    comm.Gatherv(my_vel, [vel_full.reshape(-1), rc, rd, MPI.DOUBLE] if rank == 0 else None, root=0)
    comps = comm.gather(t_comp, root=0)
    comms = comm.gather(t_comm, root=0)

    if rank == 0:
        k1, w1 = energy(pos, vel_full, mass, a.eps)
        drift_rel = ((k1 + w1) - (k0 + w0)) / abs(k0 + w0)
        mean_comp, mean_comm = float(np.mean(comps)), float(np.mean(comms))
        RESULTS.mkdir(exist_ok=True)
        np.savez(RESULTS / f"final_{rid}.npz", pos=pos, vel=vel_full, mass=mass, label=label)
        if frames:
            np.savez(RESULTS / f"frames_{rid}.npz", frames=np.array(frames))
        with open(RESULTS / f"rank_times_{rid}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "bodies", "t_comp", "t_comm"])
            for r in range(P):
                w.writerow([r, counts[r], f"{comps[r]:.6f}", f"{comms[r]:.6f}"])
        path = RESULTS / "runs.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLUMNS)
            w.writerow([rid, "mpi", a.n, P, a.steps, a.dt, a.eps, a.seed,
                        f"{t_total:.6f}", f"{mean_comp:.6f}", f"{mean_comm:.6f}",
                        os.environ.get("HW_TAG", socket.gethostname())])
        print(f"N={a.n} P={P} steps={a.steps} t_total={t_total:.3f}s comp={mean_comp:.3f}s "
              f"comm={mean_comm:.3f}s imbalance={max(comps) / mean_comp:.3f} "
              f"energy_drift={drift_rel:.2e} comm_mode={a.comm_mode} "
              f"logical_payload_per_step={a.n * (24 if a.comm_mode == 'positions' else 56)}B "
              f"run_id={rid}")


if __name__ == "__main__":
    main()