"""Distance between the two cluster centres vs time, from a run made with --save-every K."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("run_id")
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--save-every", type=int, required=True)
a = ap.parse_args()

frames = np.load(RESULTS / f"frames_{a.run_id}.npz")["frames"]  # (F, N, 3)
label = np.load(RESULTS / f"final_{a.run_id}.npz")["label"]
sep = np.linalg.norm(frames[:, label == 1].mean(axis=1) - frames[:, label == 0].mean(axis=1), axis=1)
t = np.arange(len(sep)) * a.save_every * a.dt
print(f"start sep={sep[0]:.2f}  min sep={sep.min():.2f} at t={t[sep.argmin()]:.1f}  end sep={sep[-1]:.2f}")
plt.plot(t, sep)
plt.xlabel("time (N-body units)")
plt.ylabel("distance between cluster centres")
out = RESULTS / f"separation_{a.run_id}.png"
plt.savefig(out, dpi=120)
print("saved", out)