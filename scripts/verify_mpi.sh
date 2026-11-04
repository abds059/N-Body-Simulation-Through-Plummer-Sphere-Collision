#!/usr/bin/env bash
# Run from nbody/: bash scripts/verify_mpi.sh
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NBODY_KERNEL=numba HW_TAG="${HW_TAG:-wsl}"

for n in 1000 1001; do
  sid="verify_seq_${n}"
  python src/sequential.py --n "$n" --steps 100 --run-id "$sid"
  for p in 5 7 8; do
    mid="verify_mpi_${n}_p${p}"
    mpirun --use-hwthread-cpus -np "$p" python src/mpi_nbody.py \
      --n "$n" --steps 100 --run-id "$mid"
    python src/compare.py "results/final_${sid}.npz" "results/final_${mid}.npz" --tol 1e-9
  done
done
for n in 10 20 50; do
  sid="verify_seq_${n}"
  mid="verify_mpi_${n}_p8"
  python src/sequential.py --n "$n" --steps 100 --run-id "$sid"
  mpirun --use-hwthread-cpus -np 8 python src/mpi_nbody.py \
    --n "$n" --steps 100 --run-id "$mid"
  python src/compare.py "results/final_${sid}.npz" "results/final_${mid}.npz" --tol 1e-9
done
printf '\nAll verification comparisons passed at tol=1e-9.\n'
