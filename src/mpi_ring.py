"""Optional ring-exchange MPI N-body variant (direct all-pairs, no Barnes-Hut).

Each rank owns its local positions/velocities. Remote positions travel around a
ring with MPI.Sendrecv while each rank accumulates gravitational force. Masses
are reproducibly initialized on all ranks and stay constant. This is optional:
compare numerical agreement and timings with mpi_nbody.py before reporting.
"""
import argparse
import csv
import os
import socket
import sys
import time

import numpy as np
from mpi4py import MPI
from numba import njit

from decomposition import decompose
from integrators import drift, kick
from physics import energy
from plummer import two_clusters
from sequential import COLUMNS, RESULTS


@njit(cache=True)
def force_block(target, source, source_mass, eps):
    """Acceleration on all local targets from one visiting source block."""
    out = np.zeros((len(target), 3))
    eps2 = eps * eps
    for i in range(len(target)):
        ax = ay = az = 0.0
        for j in range(len(source)):
            dx = source[j, 0] - target[i, 0]
            dy = source[j, 1] - target[i, 1]
            dz = source[j, 2] - target[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            f = source_mass[j] / (r2 * np.sqrt(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


def ring_acceleration(local_pos, mass, counts, displs, comm, buffers, eps):
    """Compute direct acceleration using P-1 Sendrecv rotations.

    Return acceleration, elapsed compute time and elapsed MPI exchange time.
    Computation includes NumPy array addition, as in the default MPI version.
    """
    rank, P = comm.Get_rank(), comm.Get_size()
    owner = rank
    block = local_pos
    total = np.zeros_like(local_pos)
    comp = comm_time = 0.0
    for rotation in range(P):
        t = MPI.Wtime()
        src_mass = mass[displs[owner]:displs[owner] + counts[owner]]
        total += force_block(local_pos, block, src_mass, eps)
        comp += MPI.Wtime() - t
        if rotation != P - 1:
            next_owner = (owner - 1) % P
            received = buffers[next_owner]
            t = MPI.Wtime()
            comm.Sendrecv([block, MPI.DOUBLE], dest=(rank + 1) % P, sendtag=17,
                          recvbuf=[received, MPI.DOUBLE], source=(rank - 1) % P,
                          recvtag=17)
            comm_time += MPI.Wtime() - t
            owner, block = next_owner, received
    return total, comp, comm_time


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
    rid = comm.bcast(a.run_id or f"ring_N{a.n}_P{P}_s{a.steps}_{int(time.time())}", root=0)
    pos, vel, mass, label = two_clusters(a.n, a.seed, a.sep, a.vfrac, a.impact)
    counts, displs = decompose(a.n, P)
    s, e = displs[rank], displs[rank] + counts[rank]
    rc, rd = [3 * c for c in counts], [3 * d for d in displs]
    local_pos, local_vel = pos[s:e].copy(), vel[s:e].copy()
    buffers = [np.empty((c, 3)) for c in counts]
    save = bool(a.save_every) and rank == 0
    frames = [pos.copy()] if save else []
    if rank == 0:
        k0, w0 = energy(pos, vel, mass, a.eps)
    # Compile the ring kernel before timing, just like main MPI's warm-up.
    force_block(pos[:2], pos[:2], mass[:2], a.eps)

    t_comp = t_comm = 0.0
    comm.Barrier()
    t0 = MPI.Wtime()
    acc, tc, tm = ring_acceleration(local_pos, mass, counts, displs, comm, buffers, a.eps)
    t_comp += tc
    t_comm += tm
    for step in range(1, a.steps + 1):
        t = MPI.Wtime()
        kick(local_vel, acc, a.dt * 0.5)
        drift(local_pos, local_vel, a.dt)
        t_comp += MPI.Wtime() - t
        acc, tc, tm = ring_acceleration(local_pos, mass, counts, displs, comm, buffers, a.eps)
        t_comp += tc
        t_comm += tm
        t = MPI.Wtime()
        kick(local_vel, acc, a.dt * 0.5)
        t_comp += MPI.Wtime() - t
        if a.save_every and step % a.save_every == 0:
            # Showcase only: a gather of positions for frames, disabled in benchmarks.
            complete = np.empty((a.n, 3)) if rank == 0 else None
            t = MPI.Wtime()
            comm.Gatherv(local_pos,
                         [complete.reshape(-1), rc, rd, MPI.DOUBLE] if rank == 0 else None,
                         root=0)
            t_comm += MPI.Wtime() - t
            if save:
                frames.append(complete)
    comm.Barrier()
    t_total = MPI.Wtime() - t0

    final_pos = np.empty((a.n, 3)) if rank == 0 else None
    final_vel = np.empty((a.n, 3)) if rank == 0 else None
    comm.Gatherv(local_pos,
                 [final_pos.reshape(-1), rc, rd, MPI.DOUBLE] if rank == 0 else None, root=0)
    comm.Gatherv(local_vel,
                 [final_vel.reshape(-1), rc, rd, MPI.DOUBLE] if rank == 0 else None, root=0)
    comps = comm.gather(t_comp, root=0)
    comms = comm.gather(t_comm, root=0)

    if rank == 0:
        k1, w1 = energy(final_pos, final_vel, mass, a.eps)
        drift_rel = ((k1 + w1) - (k0 + w0)) / abs(k0 + w0)
        mean_comp, mean_comm = float(np.mean(comps)), float(np.mean(comms))
        RESULTS.mkdir(exist_ok=True)
        np.savez(RESULTS / f"final_{rid}.npz", pos=final_pos, vel=final_vel,
                 mass=mass, label=label)
        if frames:
            np.savez(RESULTS / f"frames_{rid}.npz", frames=np.array(frames))
        with (RESULTS / f"rank_times_{rid}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "bodies", "t_comp", "t_comm"])
            for r in range(P):
                w.writerow([r, counts[r], f"{comps[r]:.6f}", f"{comms[r]:.6f}"])
        path = RESULTS / "runs.csv"
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if path.stat().st_size == 0:
                w.writerow(COLUMNS)
            w.writerow([rid, "mpi", a.n, P, a.steps, a.dt, a.eps, a.seed,
                        f"{t_total:.6f}", f"{mean_comp:.6f}", f"{mean_comm:.6f}",
                        os.environ.get("HW_TAG", socket.gethostname())])
        print(f"N={a.n} P={P} steps={a.steps} t_total={t_total:.3f}s "
              f"comp={mean_comp:.3f}s comm={mean_comm:.3f}s "
              f"imbalance={max(comps)/mean_comp:.3f} energy_drift={drift_rel:.2e} "
              f"mode=ring run_id={rid}")


if __name__ == "__main__":
    main()
