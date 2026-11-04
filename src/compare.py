"""Compare two final_<id>.npz files: max position/velocity difference and PASS/FAIL."""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--tol", type=float, default=1e-9)
args = ap.parse_args()
A, B = np.load(args.a), np.load(args.b)
if A["pos"].shape != B["pos"].shape:
    sys.exit("error: the two runs have different shapes (different N?)")
dp = np.abs(A["pos"] - B["pos"]).max()
dv = np.abs(A["vel"] - B["vel"]).max()
ok = dp <= args.tol and dv <= args.tol
print(f"max|dpos|={dp:.3e} max|dvel|={dv:.3e} tol={args.tol:.1e} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)