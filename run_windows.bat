@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Virtual environment missing. Run: py -3.12 -m venv .venv
  exit /b 1
)
set OMP_NUM_THREADS=1
set NUMBA_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
rem Windows Application Control may block Numba DLLs; use NumPy for local checks.
set NBODY_KERNEL=numpy
.venv\Scripts\python.exe scripts\test_local.py
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe src\sequential.py --n 100 --steps 50 --run-id seq_quick
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe src\simulated_mpi.py --n 100 --ranks 7 --steps 50 --run-id virtual_quick
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe src\compare.py results\final_seq_quick.npz results\final_virtual_quick.npz --tol 1e-9
endlocal
