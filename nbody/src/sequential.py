"""Sequential direct O(N^2) N-body run: the correctness and speed baseline."""
import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

from integrators import drift, kick
from physics import accel, energy
from plummer import two_clusters

RESULTS = Path(__file__).resolve().parent.parent / "results"
COLUMNS = ["run_id", "impl", "N", "P", "steps", "dt", "eps", "seed",
           "t_total", "t_comp", "t_comm", "hardware_tag"]


def main():
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
    if a.n < 2 or a.steps < 1 or a.dt <= 0 or a.eps <= 0 or a.save_every < 0:
        sys.exit("error: need n >= 2, steps >= 1, dt > 0, eps > 0, save-every >= 0")

    pos, vel, mass, label = two_clusters(a.n, a.seed, a.sep, a.vfrac, a.impact)
    k0, w0 = energy(pos, vel, mass, a.eps)
    frames = [pos.copy()] if a.save_every else []

    t0 = time.perf_counter()
    acc = accel(pos, mass, 0, a.n, a.eps)
    for step in range(1, a.steps + 1):
        kick(vel, acc, 0.5 * a.dt)
        drift(pos, vel, a.dt)
        acc = accel(pos, mass, 0, a.n, a.eps)
        kick(vel, acc, 0.5 * a.dt)
        if a.save_every and step % a.save_every == 0:
            frames.append(pos.copy())
    t_total = time.perf_counter() - t0

    k1, w1 = energy(pos, vel, mass, a.eps)
    drift_rel = ((k1 + w1) - (k0 + w0)) / abs(k0 + w0)

    rid = a.run_id or f"seq_N{a.n}_s{a.steps}_{int(time.time())}"
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"final_{rid}.npz", pos=pos, vel=vel, mass=mass, label=label)
    if frames:
        np.savez(RESULTS / f"frames_{rid}.npz", frames=np.array(frames))
    path = RESULTS / "runs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([rid, "seq", a.n, 1, a.steps, a.dt, a.eps, a.seed,
                    f"{t_total:.6f}", f"{t_total:.6f}", "0.000000",
                    os.environ.get("HW_TAG", socket.gethostname())])
    print(f"N={a.n} steps={a.steps} t_total={t_total:.3f}s "
          f"energy_drift={drift_rel:.2e} run_id={rid}")


if __name__ == "__main__":
    main()