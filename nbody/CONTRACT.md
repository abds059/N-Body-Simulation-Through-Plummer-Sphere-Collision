# CONTRACT (draft by A, review before changing)

Units: N-body units, G = 1, total mass = 1, Plummer scale radius a = 1.
Arrays: pos (N,3), vel (N,3), mass (N,), label (N,) int8 (0 = cluster 1, 1 = cluster 2).
        All float64, C-contiguous. Cluster 1 bodies come first.
Force:  accel(pos_all, mass_all, start, end, eps) -> (end-start, 3). eps must be > 0.
        Sequential calls (0, N); each MPI rank calls its own slice.
Step:   leapfrog kick-drift-kick, one accel call per step, acceleration carried to next step.
Flags:  --n --steps --dt --eps --seed --sep --vfrac --impact --save-every K (0 = off) --run-id
Files (results/):
  runs.csv  run_id, impl (seq|mpi), N, P, steps, dt, eps, seed, t_total, t_comp, t_comm, hardware_tag
  final_<id>.npz   pos, vel, mass, label
  frames_<id>.npz  frames (positions every K steps; showcase runs only)
Timing: covers the initial accel plus all steps; no I/O, plotting or energy checks inside it.