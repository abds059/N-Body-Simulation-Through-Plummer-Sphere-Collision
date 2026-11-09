
# Hardware and Software Environment

This file records the environment used for the final CT-488 Distributed N-Body Simulation experiments on **9 November 2026**. The values below were taken from `results/cpu_info.txt`, `results/memory_info.txt`, and `results/system_info.txt` saved with the experiment logs.

## Hardware

| Component | Recorded configuration |
|---|---|
| Processor | AMD Ryzen 5 7430U with Radeon Graphics |
| Architecture | x86-64 |
| Physical CPU cores | 6 |
| Logical CPU threads | 12 (2 threads per core) |
| CPU sockets | 1 |
| L3 cache | 16 MiB |
| Memory visible to WSL2 | 3.5 GiB |
| Swap visible to WSL2 | 1.0 GiB |
| Execution platform | Windows host with Linux running under WSL2 |

**Note:** The memory values are limits visible *inside WSL2*, not the total physical RAM installed in the Windows host. A separate host-RAM measurement was not included in the experiment logs. The GPU was not used for computing the gravitational forces.

## Software

| Component | Version / configuration |
|---|---|
| Python | 3.14.4 (GCC 15.2.0 build) |
| NumPy | 2.5.4 |
| Numba | 0.68.1 |
| mpi4py | 4.1.2 |
| Open MPI | 5.0.10 |
| Matplotlib | 3.11.2 |
| Linux kernel | 6.18.54.1-microsoft-standard-WSL2 |
| MPI launcher | `/usr/bin/mpirun` |
| Force kernel for benchmark | `NBODY_KERNEL=numba` |

The Linux distribution name/version was not captured in the saved logs and is therefore not specified here.

## Benchmark Configuration

| Setting | Recorded value |
|---|---|
| Main experiment date | 2026-11-09 |
| Hardware tag in `runs.csv` | `wsl_cca_20261109` |
| Benchmark run prefix | `cca_20261109_133748` |
| Body counts (N) | 100, 500, 1,000, 5,000, 10,000 |
| MPI process counts (P) | 1, 2, 4, 8 |
| Time steps per benchmark | 100 |
| Repeats per configuration | 3 |
| Random seed | 42 |
| Integration time step (`dt`) | 0.01 |
| Gravitational softening (`eps`) | 0.05 |
| Numerical comparison tolerance | 1e-9 (maximum absolute position and velocity differences) |
| MPI mapping for benchmarks | Open MPI launched with `--use-hwthread-cpus` |

MPI rank counts of 1, 2 and 4 were within the six physical CPU cores; **P = 8 exceeded the physical-core count but remained within the 12 logical threads**, so the eight-rank results also reflect hardware-thread sharing. All benchmark measurements were performed on this single WSL2 environment.

For reproducibility, see `README.md`. The raw timing measurements and validation evidence are saved in `results/runs.csv`, `results/cca_20261109_133748_grid.log`, `results/rank_times_*.csv`, and `results/plots/`.
