"""Interactive N-body demo; simulation jobs launch separately from the UI.

The Plummer collision is the graded benchmark scenario. The extra planetary
mode is an illustrative toy and not a source of MPI performance measurements.
"""
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from plummer import two_clusters  # noqa: E402
from space_view import planetary_frames, space_figure  # noqa: E402

st.set_page_config(page_title="N-Body Space Laboratory", page_icon="🌌", layout="wide")
st.title("🌌 N-Body Space Laboratory")
st.caption("Real numerical gravity · 3D interactive playback · sequential / MPI comparison")


def run_process(command, timeout=480):
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NBODY_KERNEL": env.get("NBODY_KERNEL", "numpy" if os.name == "nt" else "numba")})
    return subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)


@st.cache_data(show_spinner=False)
def cached_planets(steps, dt, save_every):
    return planetary_frames(steps, dt, save_every)


with st.sidebar:
    st.header("Explore space")
    mode = st.radio("Choose a simulation", ["Star clusters (course project)",
                                            "Planetary orbits (bonus demo)"])
    st.divider()
    if mode.startswith("Star"):
        st.header("Collision parameters")
        n = st.number_input("Stars / bodies (N)", min_value=2, max_value=2500, value=120, step=20)
        steps = st.number_input("Simulation steps", min_value=1, max_value=4000, value=1650, step=50)
        save_every = st.number_input("Save frame every K steps", min_value=1, max_value=500,
                                     value=30, step=5)
        dt = st.number_input("Time step (dt)", min_value=0.0001, max_value=0.1,
                             value=0.01, format="%.4f")
        eps = st.number_input("Softening (eps)", min_value=0.0001, max_value=1.0,
                              value=0.05, format="%.4f")
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
        sep = st.number_input("Initial separation", min_value=1.0, max_value=30.0,
                              value=8.0, step=0.5)
        vfrac = st.number_input("Relative speed fraction", min_value=0.0,
                                max_value=2.0, value=0.6, step=0.1)
        impact = st.number_input("Impact parameter", min_value=0.0,
                                 max_value=10.0, value=1.0, step=0.25)
        p = st.selectbox("MPI processes (P)", [1,2,3,4,5,7,8], index=2)
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        do_mpi = st.checkbox("Run real MPI too", value=False, disabled=not bool(launcher))
        if not launcher:
            st.caption("MPI not installed: the sequential collision and 3D animation still work.")
        st.caption("For a visible collision, use about 1,600+ steps with dt=0.01."
                   " Small N is faster on Windows without Numba.")
        run_button = st.button("🚀 Simulate collision", type="primary", use_container_width=True)
    else:
        st.header("Planetary settings")
        orbit_steps = st.slider("Orbital simulation steps", 500, 3500, 2300, 100)
        orbit_dt = st.slider("Orbital time step", 0.005, 0.025, 0.015, 0.005)
        orbit_save = st.slider("Save every K steps", 5, 40, 15, 5)
        st.caption("Fictional planets, illustrative sizes and colors, G=1."
                   " Computed with the same direct-gravity / leapfrog principle."
                   " This is not the required two-cluster benchmark.")

if mode.startswith("Planetary"):
    st.subheader("🪐 Interactive orbital system")
    st.write("A massive central star and four orbiting bodies attract each other."
             " The planets move using calculated gravitational forces rather than a drawn animation.")
    with st.spinner("Calculating planetary orbits"):
        orbits, planet_labels = cached_planets(orbit_steps, orbit_dt, orbit_save)
    st.plotly_chart(space_figure(orbits, planet_labels, mode="planets",
                                 title="Toy Five-Body Orbital Simulation"),
                    use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})
    st.info("Rotate the 3D view with the mouse. Use ▶ Play or drag the frame slider beneath the scene."
            " Orbit paths are based on simulated positions; body sizes are enlarged for visibility.")
    st.stop()

if run_button:
    tag = datetime.now().strftime("gui_%Y%m%d_%H%M%S_%f")
    common = ["--n", str(n), "--steps", str(steps), "--dt", str(dt),
              "--eps", str(eps), "--seed", str(seed), "--sep", str(sep),
              "--vfrac", str(vfrac), "--impact", str(impact),
              "--save-every", str(save_every)]
    sid = f"{tag}_seq"
    with st.spinner("Computing the real N-body star-cluster collision..."):
        try:
            a = run_process([sys.executable, str(ROOT / "src/sequential.py"),
                             *common, "--run-id", sid])
        except subprocess.TimeoutExpired:
            st.error("Sequential simulation timed out. Try a smaller N.")
            st.stop()
    with st.expander("Sequential terminal output"):
        st.code(a.stdout)
    if a.returncode:
        st.error(f"Sequential simulation failed (exit {a.returncode}).")
        st.stop()

    mid = None
    if do_mpi and launcher and p <= n:
        mid = f"{tag}_mpi"
        cmd = [launcher]
        if os.name != "nt" and p > 4:
            cmd.append("--use-hwthread-cpus")
        cmd += ["-np", str(p), sys.executable, str(ROOT / "src/mpi_nbody.py"),
                *common, "--run-id", mid]
        with st.spinner("Running the MPI version..."):
            try:
                b = run_process(cmd, timeout=600)
            except subprocess.TimeoutExpired:
                st.error("MPI timed out")
                st.stop()
        with st.expander("MPI terminal output"):
            st.code(b.stdout)
        if b.returncode:
            st.error("MPI launch failed; check launcher / mpi4py installation.")
            mid = None
    elif do_mpi and p > n:
        st.warning("MPI requires N >= P")
    st.session_state["last_seq"] = sid
    st.session_state["last_mpi"] = mid
    st.session_state["last_settings"] = {"dt": dt, "save_every": save_every,
                                          "steps": steps, "n": n}

sid = st.session_state.get("last_seq")
mid = st.session_state.get("last_mpi")
if sid and (RESULTS / f"final_{sid}.npz").exists():
    data = np.load(RESULTS / f"final_{sid}.npz")
    path = RESULTS / f"frames_{sid}.npz"
    if path.exists():
        frames = np.load(path)["frames"]
        st.subheader("✨ Star-cluster collision — 3D playback")
        st.caption(f"{len(frames)} saved frames. Blue and orange represent stars"
                   " belonging to the two original clusters."
                   " The faint background stars are decoration, not simulated bodies.")
        st.plotly_chart(space_figure(frames, data["label"], mode="clusters"),
                        use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False})
        st.info("Use ▶ Play or the frame slider **inside the 3D scene** to watch the collision."
                " Drag to rotate, scroll to zoom.")
        if len(frames) > 1:
            settings = st.session_state.get("last_settings", {})
            every = settings.get("save_every", 1)
            step_dt = settings.get("dt", .01)
            ca = frames[:, data["label"] == 0].mean(axis=1)
            cb = frames[:, data["label"] == 1].mean(axis=1)
            dist = np.linalg.norm(ca - cb, axis=1)
            f, ax = plt.subplots(figsize=(9, 2.7))
            ax.plot(np.arange(len(frames)) * every * step_dt, dist)
            ax.set(xlabel="Simulation time", ylabel="Center separation",
                   title="Distance between the two star clusters")
            f.tight_layout()
            st.pyplot(f)
            plt.close(f)
    else:
        st.warning("Only the final coordinates were saved; rerun with Save every K > 0 to animate.")
        st.plotly_chart(space_figure(data["pos"], data["label"], mode="clusters"),
                        use_container_width=True)
    if mid and (RESULTS / f"final_{mid}.npz").exists():
        other = np.load(RESULTS / f"final_{mid}.npz")
        dp = float(np.max(np.abs(data["pos"] - other["pos"])))
        dv = float(np.max(np.abs(data["vel"] - other["vel"])))
        ok = max(dp, dv) <= 1e-9
        st.subheader("Sequential versus real MPI")
        st.write(f"Max position difference: `{dp:.3e}`; max velocity difference: `{dv:.3e}`")
        if ok:
            st.success("PASS: within 1e-9 numerical tolerance")
        else:
            st.error("FAIL: differences exceed the tolerance")
        with (RESULTS / "runs.csv").open(newline="") as f:
            selected = [r for r in csv.DictReader(f) if r["run_id"] in {sid, mid}]
        st.dataframe(selected, use_container_width=True)
        timings = {r["impl"]: float(r["t_total"]) for r in selected}
        if "seq" in timings and "mpi" in timings and timings["mpi"] > 0:
            speed = timings["seq"] / timings["mpi"]
            c1, c2 = st.columns(2)
            c1.metric("Live run speedup", f"{speed:.2f}×")
            c2.metric("Live run efficiency", f"{speed/p:.1%}")
            st.caption("Single demonstration only, not a repeat-based benchmark.")
else:
    st.subheader("🌠 Initial star-cluster positions")
    st.write("This is the starting configuration. Click **Simulate collision** in the sidebar"
             " to calculate gravitational motion and enable 3D playback.")
    preview, _, _, labels = two_clusters(n, seed, sep, vfrac, impact)
    st.plotly_chart(space_figure(preview, labels, mode="clusters",
                                 title="Initial positions — not yet animated"),
                    use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})

st.divider()
st.subheader("Saved benchmark results")
if (RESULTS / "runs.csv").exists():
    with (RESULTS / "runs.csv").open(newline="") as f:
        stored = list(csv.DictReader(f))
    st.dataframe(stored[-50:][::-1], use_container_width=True)
else:
    st.write("No saved experiments yet.")
