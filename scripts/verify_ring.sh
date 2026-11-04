#!/usr/bin/env bash
# Run from nbody/: bash scripts/verify_ring.sh
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NBODY_KERNEL=numba HW_TAG="${HW_TAG:-wsl}"
for n in 100 1001; do
  python src/sequential.py --n "$n" --steps 100 --run-id "ring_test_seq_${n}"
  for p in 1 2 3 4; do
    mpirun --use-hwthread-cpus -np "$p" python src/mpi_ring.py \
      --n "$n" --steps 100 --run-id "ring_test_${n}_${p}"
    python src/compare.py "results/final_ring_test_seq_${n}.npz" \
      "results/final_ring_test_${n}_${p}.npz" --tol 1e-9
  done
done
printf '\nRing test comparisons passed at tol=1e-9.\n'
