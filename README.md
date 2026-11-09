
# Distributed N-Body Simulation — Plummer Sphere Collision

**CT-488 Distributed Computing | Department of Computer Science & Information Technology**

A simulation of two colliding star clusters using **direct O(N²) gravitational interactions** and **leapfrog integration**. The repository includes a sequential reference implementation, a distributed MPI implementation, correctness checks, repeatable benchmarks, visualizations, and an optional 3D web interface.

The main experiment uses **two Plummer-sphere star clusters**, not planets. The GUI also offers a separate, illustrative planetary-orbit demo; it is **not** part of the distributed benchmark.

## 1. Project structure

```text
src/
  sequential.py         Sequential direct N-body baseline
  mpi_nbody.py          Distributed MPI implementation (Allgatherv)
  simulated_mpi.py      Virtual-rank numerical check (one process; not MPI)
  mpi_ring.py           Optional MPI ring-communication variant
  physics.py            NumPy force kernel / kernel selection
  physics_numba.py      Numba-accelerated force kernel
  plummer.py            Initial conditions for two star clusters
  integrators.py        Leapfrog integration helpers
  decomposition.py      Block partitioning across ranks
  compare.py            Numerical agreement checker
  loadbalance.py        Per-rank workload table and chart
  costmodel.py          Exploratory performance model
  diagnostics.py        Cluster-separation graph
  viz.py                2D collision image and optional GIF
  test_baseline.py      Baseline physics tests
  test_decomposition.py Decomposition tests
scripts/
  test_local.py         Local correctness suite; no MPI required
  verify_mpi.py         Real MPI numerical verification
  verify_ring.py        Ring-variant verification
  run_grid.py           Repeated sequential / MPI experiment launcher
  make_plots.py         Performance summary CSV and graphs
  comm_compare.py       Positions-only vs full-state MPI comparison
  system_info.py        Software and system information
  *.sh                  Linux shell wrappers for MPI checks
gui/
  app.py                Streamlit web interface
  space_view.py         Interactive 3D visualization helpers
results/                 Generated CSVs, final states, frames, and figures
CONTRACT.md              Shared data formats and conventions
HARDWARE.md              Hardware information for benchmark records
requirements*.txt        Python dependencies
run_windows.bat          Windows local-test launcher
```

Run commands from the **project root** (the folder containing `src/`, `scripts/`, and `requirements.txt`).

## 2. Installation

Python **3.12** is recommended. Use one virtual environment for the project.

### Windows — local simulation and GUI (MPI not required)

In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
$env:NBODY_KERNEL = "numpy"
```

The NumPy kernel is suitable for local testing and visualizations. On computers where Windows Application Control blocks Numba DLLs, keep `NBODY_KERNEL=numpy`. Numba is faster when available, but is not necessary to use the GUI or local numerical checks.

### Linux / WSL2 — real MPI experiments

Install Python tools and Open MPI in Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential openmpi-bin libopenmpi-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mpi.txt
python -m pip install -r requirements-gui.txt   # optional GUI
```

Set single-threaded execution and select the accelerated kernel:

```bash
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NBODY_KERNEL=numba
export HW_TAG=wsl
```

Check that MPI can launch processes:

```bash
mpirun -np 2 python -c "from mpi4py import MPI; print('Rank', MPI.COMM_WORLD.Get_rank())"
python scripts/system_info.py
```

**Windows real MPI is also possible** with Microsoft MPI and `mpi4py`. Use `mpiexec` instead of `mpirun`; the MPI installation and Numba native libraries must work in that environment. The provided benchmark and MPI verification launchers select **Numba** for their subprocesses, so a Windows installation where Numba is blocked cannot run those scripts unchanged.

## 3. Quick start — no MPI needed

### Run all local checks on Windows

```powershell
.\run_windows.bat
```

This runs the local numerical checks, a sequential simulation, a virtual-rank simulation, and a `1e-9` comparison. It does **not** run multiple MPI processes.

For individual commands, after setting `NBODY_KERNEL=numpy`:

```powershell
.\.venv\Scripts\python.exe scripts\test_local.py
.\.venv\Scripts\python.exe src\sequential.py --n 200 --steps 100 --run-id demo_seq
.\.venv\Scripts\python.exe src\simulated_mpi.py --n 200 --ranks 4 --steps 100 --run-id demo_virtual
.\.venv\Scripts\python.exe src\compare.py results\final_demo_seq.npz results\final_demo_virtual.npz --tol 1e-9
```

`simulated_mpi.py` imitates body partitioning in **one process**, to validate the algorithm. Do not use its timings as distributed performance measurements.

## 4. Launch the 3D GUI

From the project root, with GUI dependencies installed:

**Windows PowerShell:**

```powershell
$env:NBODY_KERNEL = "numpy"
.\.venv\Scripts\python.exe -m streamlit run gui\app.py
```

**Linux / WSL2:**

```bash
python -m streamlit run gui/app.py
```

Open **http://localhost:8501** if your browser does not open automatically.

The sidebar lets you select:

- **Star clusters (course project):** Set N, steps, time step, softening, seed, and collision settings. Click **Simulate collision** to view the actual computed motion in interactive 3D, with Play/Pause and a frame slider. When a working MPI launcher is detected, you can also select **Run real MPI too**.
- **Planetary orbits (bonus demo):** An independent five-body gravitational demonstration. It is for visualization only and is not used in MPI benchmarks.

**Tip:** At `dt=0.01`, a visible cluster encounter typically needs around **1,600–1,800 steps**. Use a small N (for example, 120) when running with the slower NumPy kernel on Windows. A 100-step run is better suited to benchmarking than watching the complete collision.

## 5. Run the sequential and MPI versions

Use the following commands in **Linux / WSL2**, after activating the environment and setting the variables in Section 2.

### Sequential reference

```bash
python src/sequential.py --n 1000 --steps 100 --run-id seq_1000
```

### Real MPI — 4 processes

```bash
mpirun -np 4 python src/mpi_nbody.py --n 1000 --steps 100 --run-id mpi_1000_p4
```

### Real MPI — 8 processes

```bash
mpirun --use-hwthread-cpus -np 8 python src/mpi_nbody.py --n 1000 --steps 100 --run-id mpi_1000_p8
```

The `--use-hwthread-cpus` flag is for **Open MPI** when hardware threads must be included as available slots. On machines without enough slots, consult the MPI runtime's process-mapping settings. Runs using more ranks than physical CPU cores must be identified when interpreting scaling.

Compare the final states:

```bash
python src/compare.py results/final_seq_1000.npz results/final_mpi_1000_p4.npz --tol 1e-9
```

The comparison checks the maximum absolute differences in **both positions and velocities**. It prints `PASS` only when each is at most `1e-9`.

The main MPI implementation uses block decomposition. When `N` is not divisible by `P`, the first `N % P` ranks receive one additional body. Each rank computes its own bodies, while updated positions are exchanged through one `MPI_Allgatherv` per simulation step.

## 6. Simulation parameters

The sequential and MPI commands share these options:

| Option | Default | Description |
|---|---:|---|
| `--n` | `1000` | Number of bodies |
| `--steps` | `100` | Number of integration steps |
| `--dt` | `0.01` | Time increment |
| `--eps` | `0.05` | Gravitational softening length |
| `--seed` | `42` | Reproducible initial conditions |
| `--sep` | `8.0` | Initial cluster separation |
| `--vfrac` | `0.6` | Relative approach-speed factor |
| `--impact` | `1.0` | Off-center collision offset |
| `--save-every` | `0` | Save positions every K steps; `0` disables frames |
| `--run-id` | automatic | Identifier used in output filenames |
| `--comm-mode` | `positions` | MPI only: `positions` or experimental `full-state` |
| `--ranks` | `4` | Virtual-rank simulator only; not an MPI process count |

The simulation uses normalized N-body units (`G=1`, total mass `1`, Plummer radius `1`). `N` must be at least the number of MPI ranks, and `dt` and `eps` must be positive.

## 7. Tests and verification

**Local, without MPI:**

```bash
python scripts/test_local.py
python src/test_decomposition.py
python src/test_baseline.py
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `python` and set `NBODY_KERNEL=numpy` first.

**Real MPI correctness, once an MPI runtime is installed:**

```bash
python scripts/verify_mpi.py --quick
python scripts/verify_mpi.py --hwthreads
```

The full verification script includes uneven workloads and process counts up to 8. Omit `--hwthreads` for Microsoft MPI; use it only when appropriate for Open MPI. These commands launch the sequential and MPI programs and check the saved results using the same numerical tolerance.

Optional ring-variant verification:

```bash
python scripts/verify_ring.py --quick
```

## 8. Benchmarks and performance plots

The experiment launcher runs each configuration in a fresh process and checks MPI results against the sequential baseline. Use **one machine, one kernel, and the same settings** for all timings in a comparison.

**Full experiment grid (real MPI required):**

```bash
python scripts/run_grid.py --n 100 500 1000 5000 10000 --p 1 2 4 8 --steps 100 --repeats 3 --hwthreads --tag grid_
python scripts/make_plots.py --prefix grid_
```

Omit `--hwthreads` when using Microsoft MPI or when it is not needed on your Open MPI machine. The launcher currently sets `NBODY_KERNEL=numba`, so this command requires functioning Numba binaries. A run at N=10,000 may be significantly more expensive than small test cases.

**Sequential-only benchmark (Numba still required by this launcher):**

```bash
python scripts/run_grid.py --no-mpi --n 100 500 1000 --steps 100 --repeats 3 --tag seq_only_
```

**Generate plots for existing measurements:**

```bash
python scripts/make_plots.py --prefix grid_ --hardware-tag wsl
```

The hardware filter is optional. Replace `wsl` with the `hardware_tag` actually recorded in your CSV; do not combine different machines in one speedup comparison.

**Per-rank load balance** (after an MPI run):

```bash
python src/loadbalance.py mpi_1000_p4
```

For uneven partitioning, inspect an MPI run such as N=1001 and P=4 and use that run's ID instead.

**Communication experiment** (real MPI required):

```bash
python scripts/comm_compare.py --n 2000 --p 4 --steps 100 --repeats 3
```

This compares the default positions-only communication against an intentionally larger full-state exchange. Both modes use the same physics; the extra state is sent for communication-overhead analysis only.

**Exploratory performance model** (requires multiple matching MPI benchmark cases):

```bash
python src/costmodel.py --hardware-tag wsl --run-prefix grid_ --steps 100
```

**Optional ring implementation** (real MPI required):

```bash
mpirun -np 4 python src/mpi_ring.py --n 1000 --steps 100 --run-id ring_1000_p4
python scripts/verify_ring.py --quick
```

## 9. Create collision images and animations

Save intermediate frames during a **visualization run**:

```bash
python src/sequential.py --n 200 --steps 1700 --save-every 25 --run-id collision_demo
python src/viz.py collision_demo --gif
python src/diagnostics.py collision_demo --save-every 25
```

The scripts save a 2D collision image, an animated GIF, and a graph of the cluster-center separation in `results/`. For interactive 3D playback, use the Streamlit GUI instead.

For repeatable timing measurements, use `--save-every 0` (the default) to avoid saving frames inside the simulation loop.

## 10. Output files

All generated files are stored in `results/`:

| File | Purpose |
|---|---|
| `runs.csv` | Raw run records: configuration and total / computation / communication times |
| `final_<run_id>.npz` | Final positions, velocities, masses, and cluster labels |
| `frames_<run_id>.npz` | Saved position frames when `--save-every > 0` |
| `rank_times_<run_id>.csv` | Per-rank body counts, computation time, and communication time (real MPI) |
| `loadbalance_<run_id>.png` | Measured per-rank load-balance graph |
| `collision_<run_id>.png` / `.gif` | 2D collision visualization |
| `separation_<run_id>.png` | Cluster-center separation graph |
| `costmodel.png` | Exploratory cost-model graph, if generated |
| `plots/performance_summary.csv` | Grouped timing statistics and derived speedup / efficiency |
| `plots/*.png` | Benchmark plots, including runtime, speedup, efficiency, and communication fraction when applicable |

The MPI CSV reports mean per-rank computation and communication times. Communication time includes time waiting at collective operations. Total wall-clock time is measured separately; mean compute and mean communication figures should not be assumed to sum exactly to total time.

## 11. Reproducibility and troubleshooting

- Use a **fixed random seed** and identical simulation parameters for sequential and MPI comparisons.
- Use `--repeats 3` (or more) for benchmark configurations; single GUI runs are illustrative, not final benchmark averages.
- Set `OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to avoid extra threads competing with MPI ranks.
- Record the Python, NumPy, Numba, mpi4py, and MPI versions using `python scripts/system_info.py`; document the machine in `HARDWARE.md`.
- Keep hardware tags distinct when runs come from different operating systems, machines, or runtime environments.
- **Numba DLL blocked on Windows:** set `$env:NBODY_KERNEL = "numpy"` for sequential/GUI/local tests. The benchmark and MPI verification launchers explicitly select Numba and require it to work.
- **MPI launcher not found:** install Open MPI on Linux/WSL2 or a supported MPI runtime on Windows; a virtual-rank test is not a replacement for MPI validation.
- **Insufficient MPI slots:** for Open MPI, use hardware-thread mapping where appropriate, or reduce `P`.
- **Simulation too slow on Windows:** reduce `N`, especially when using the NumPy kernel; keep large benchmark runs on a machine with working Numba.
- **No speedup graph:** first collect both sequential and MPI measurements with matching N, steps, physics settings, and hardware tag.

For file formats and implementation assumptions, see [`CONTRACT.md`](CONTRACT.md). For the benchmark machine, see [`HARDWARE.md`](HARDWARE.md).
