#!/usr/bin/env bash
# Run from nbody/: bash scripts/comm_compare.sh
# Experimental-only measurements, not final benchmark grid.
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NBODY_KERNEL=numba HW_TAG="${HW_TAG:-wsl}"

N="${1:-2000}"
P="${2:-4}"
STEPS="${3:-100}"
TAG="$(date +%s)_$$"
SEQ="comm_seq_N${N}_s${STEPS}_${TAG}"
POS="comm_pos_N${N}_P${P}_s${STEPS}_${TAG}"
FULL="comm_full_N${N}_P${P}_s${STEPS}_${TAG}"
python src/sequential.py --n "$N" --steps "$STEPS" --run-id "$SEQ"
mpirun --use-hwthread-cpus -np "$P" python src/mpi_nbody.py \
  --n "$N" --steps "$STEPS" --run-id "$POS" --comm-mode positions
mpirun --use-hwthread-cpus -np "$P" python src/mpi_nbody.py \
  --n "$N" --steps "$STEPS" --run-id "$FULL" --comm-mode full-state
python src/compare.py "results/final_${SEQ}.npz" "results/final_${POS}.npz" --tol 1e-9
python src/compare.py "results/final_${SEQ}.npz" "results/final_${FULL}.npz" --tol 1e-9
printf 'Logical payload per step: positions=%d bytes; full-state=%d bytes\n' "$((24*N))" "$((56*N))"
printf 'See results/runs.csv for observed t_comm and t_total. Repeat 3+ times for conclusions.\n'
