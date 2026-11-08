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

CUSTOM_CSS = """
<style>
:root {
  --panel:#0d1627;
  --panel2:#0a1220;
  --line:#243752;
  --text:#dce7fb;
  --muted:#9fb2d0;
  --accent:#63d5ff;
}
.stApp {
  background:
    radial-gradient(circle at 15% 20%, rgba(52, 110, 185, 0.20), transparent 20%),
    radial-gradient(circle at 85% 15%, rgba(255, 138, 76, 0.16), transparent 18%),
    radial-gradient(circle at 65% 80%, rgba(107, 84, 182, 0.14), transparent 24%),
    linear-gradient(180deg, #020611 0%, #07111e 100%);
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #07101c 0%, #091423 100%);
  border-right: 1px solid rgba(122, 156, 204, 0.12);
}
[data-testid="stMetric"] {
  background: rgba(10, 18, 32, 0.85);
  border: 1px solid rgba(90, 116, 158, 0.28);
  border-radius: 16px;
  padding: 10px 16px;
}
div[data-testid="stMarkdownContainer"] p, .stCaption { color: var(--text); }
.block-card {
  background: rgba(10, 18, 32, 0.82);
  border: 1px solid rgba(95, 124, 171, 0.24);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 14px 34px rgba(0,0,0,0.22);
}
.note-card {
  background: linear-gradient(180deg, rgba(13, 22, 39, 0.92), rgba(10, 18, 32, 0.92));
  border: 1px solid rgba(87, 126, 176, 0.28);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
}
hr { border-color: rgba(96,122,166,0.20); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🌌 Distributed N-Body Simulation")
st.caption("Real numerical star-cluster collision with playback controls and optional MPI comparison")


def run_process(command, timeout=480):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NBODY_KERNEL": env.get("NBODY_KERNEL", "numpy" if os.name == "nt" else "numba")
    })
    return subprocess.run(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=timeout
    )


@st.cache_data(show_spinner=False)
def cached_planets(steps, dt, save_every):
    return planetary_frames(steps, dt, save_every)


with st.sidebar:
    st.header("Simulation setup")
    mode = st.radio(
        "Select view",
        ["Star clusters (course project)", "Planetary demo (visual only)"],
        help="The graded project is the star-cluster collision. The planetary demo is only a visual extra."
    )
    st.divider()
    if mode.startswith("Star"):
        st.subheader("Collision parameters")
        n = st.number_input("Bodies / stars (N)", min_value=2, max_value=2500, value=120, step=20)
        steps = st.number_input("Simulation steps", min_value=1, max_value=4000, value=1650, step=50)
        save_every = st.number_input("Save frame every K steps", min_value=1, max_value=500, value=30, step=5)
        dt = st.number_input("Time step (dt)", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
        eps = st.number_input("Softening (eps)", min_value=0.0001, max_value=1.0, value=0.05, format="%.4f")
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
        sep = st.number_input("Initial separation", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
        vfrac = st.number_input("Relative speed fraction", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
        impact = st.number_input("Impact parameter", min_value=0.0, max_value=10.0, value=1.0, step=0.25)
        p = st.selectbox("MPI processes (P)", [1, 2, 3, 4, 5, 7, 8], index=2)
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        do_mpi = st.checkbox("Run MPI version too", value=False, disabled=not bool(launcher))
        if not launcher:
            st.caption("MPI launcher not found. Sequential animation still works.")
        st.caption("Tip: for a visible collision, use roughly 1,600 or more steps with dt=0.01.")
        run_button = st.button("🚀 Run simulation", type="primary", use_container_width=True)
    else:
        st.subheader("Planetary demo")
        orbit_steps = st.slider("Orbital simulation steps", 500, 3500, 2300, 100)
        orbit_dt = st.slider("Orbital time step", 0.005, 0.025, 0.015, 0.005)
        orbit_save = st.slider("Save every K steps", 5, 40, 15, 5)
        st.caption("Optional visual demo only. It is not part of the benchmarked course result.")

if mode.startswith("Planetary"):
    st.markdown(
        """
        <div class="note-card">
        <b>Optional visual demo.</b> This panel is only for showcasing the interface.
        It is not used as evidence for the distributed-computing project or the performance report.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("🪐 Interactive orbital system")
    st.write(
        "A massive central star and four orbiting bodies attract each other. "
        "The motion is computed numerically; it is not a pre-drawn animation."
    )
    with st.spinner("Calculating planetary orbits"):
        orbits, planet_labels = cached_planets(orbit_steps, orbit_dt, orbit_save)
    st.plotly_chart(
        space_figure(orbits, planet_labels, mode="planets", title="Optional Planetary Demo"),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.info("Use the play/pause controls under the scene. You can drag to rotate and scroll to zoom.")
    st.stop()

st.markdown(
    """
    <div class="block-card">
    <b>What this screen shows:</b> the real course-project visualization is the collision of two Plummer star clusters.
    Each bright point is a simulated body. The background galaxy effect is decorative only; the stars and their trails are the computed data.
    </div>
    """,
    unsafe_allow_html=True,
)

if run_button:
    tag = datetime.now().strftime("gui_%Y%m%d_%H%M%S_%f")
    common = [
        "--n", str(n), "--steps", str(steps), "--dt", str(dt),
        "--eps", str(eps), "--seed", str(seed), "--sep", str(sep),
        "--vfrac", str(vfrac), "--impact", str(impact), "--save-every", str(save_every)
    ]
    sid = f"{tag}_seq"
    with st.spinner("Computing the sequential N-body collision..."):
        try:
            a = run_process([sys.executable, str(ROOT / "src/sequential.py"), *common, "--run-id", sid])
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
        cmd += ["-np", str(p), sys.executable, str(ROOT / "src/mpi_nbody.py"), *common, "--run-id", mid]
        with st.spinner("Running the MPI version..."):
            try:
                b = run_process(cmd, timeout=600)
            except subprocess.TimeoutExpired:
                st.error("MPI timed out")
                st.stop()
        with st.expander("MPI terminal output"):
            st.code(b.stdout)
        if b.returncode:
            st.error("MPI launch failed. Check the MPI launcher and mpi4py installation.")
            mid = None
    elif do_mpi and p > n:
        st.warning("MPI requires N >= P")

    st.session_state["last_seq"] = sid
    st.session_state["last_mpi"] = mid
    st.session_state["last_settings"] = {"dt": dt, "save_every": save_every, "steps": steps, "n": n}

sid = st.session_state.get("last_seq")
mid = st.session_state.get("last_mpi")
if sid and (RESULTS / f"final_{sid}.npz").exists():
    data = np.load(RESULTS / f"final_{sid}.npz")
    path = RESULTS / f"frames_{sid}.npz"
    if path.exists():
        frames = np.load(path)["frames"]
        st.subheader("✨ Star-cluster collision playback")
        st.caption(
            f"{len(frames)} saved frames. Bright bodies are simulated stars. "
            f"Soft glows, background stars, and the galaxy backdrop are visual styling for readability."
        )
        st.plotly_chart(
            space_figure(frames, data["label"], mode="clusters"),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
        st.info(
            "Use ▶ Play or the frame slider inside the 3D scene. "
            "Hover on a body to see its cluster and coordinates."
        )
        if len(frames) > 1:
            settings = st.session_state.get("last_settings", {})
            every = settings.get("save_every", 1)
            step_dt = settings.get("dt", 0.01)
            ca = frames[:, data["label"] == 0].mean(axis=1)
            cb = frames[:, data["label"] == 1].mean(axis=1)
            dist = np.linalg.norm(ca - cb, axis=1)
            f, ax = plt.subplots(figsize=(9, 2.7))
            ax.plot(np.arange(len(frames)) * every * step_dt, dist, color="#86dcff", linewidth=2.2)
            ax.set(
                xlabel="Simulation time",
                ylabel="Center separation",
                title="Distance between the two cluster centers",
            )
            ax.set_facecolor("#07101b")
            f.patch.set_facecolor("#07101b")
            ax.tick_params(colors="#dce7fb")
            ax.xaxis.label.set_color("#dce7fb")
            ax.yaxis.label.set_color("#dce7fb")
            ax.title.set_color("#e8f0ff")
            for spine in ax.spines.values():
                spine.set_color("#38506f")
            f.tight_layout()
            st.pyplot(f)
            plt.close(f)
    else:
        st.warning("Only the final coordinates were saved. Rerun with Save every K > 0 to animate the motion.")
        st.plotly_chart(space_figure(data["pos"], data["label"], mode="clusters"), use_container_width=True)

    if mid and (RESULTS / f"final_{mid}.npz").exists():
        other = np.load(RESULTS / f"final_{mid}.npz")
        dp = float(np.max(np.abs(data["pos"] - other["pos"])))
        dv = float(np.max(np.abs(data["vel"] - other["vel"])))
        ok = max(dp, dv) <= 1e-9
        st.subheader("Sequential versus MPI")
        c1, c2 = st.columns(2)
        c1.metric("Max position difference", f"{dp:.3e}")
        c2.metric("Max velocity difference", f"{dv:.3e}")
        if ok:
            st.success("PASS: sequential and MPI outputs match within the 1e-9 tolerance.")
        else:
            st.error("FAIL: differences exceed the tolerance.")

        with (RESULTS / "runs.csv").open(newline="") as f:
            selected = [r for r in csv.DictReader(f) if r["run_id"] in {sid, mid}]
        st.dataframe(selected, use_container_width=True)
        timings = {r["impl"]: float(r["t_total"]) for r in selected}
        if "seq" in timings and "mpi" in timings and timings["mpi"] > 0:
            speed = timings["seq"] / timings["mpi"]
            m1, m2 = st.columns(2)
            m1.metric("Live run speedup", f"{speed:.2f}×")
            m2.metric("Live run efficiency", f"{speed / p:.1%}")
            st.caption("This is a single demonstration run, not the full repeat-based benchmark table.")
else:
    st.subheader("🌠 Initial star-cluster scene")
    st.write(
        "This is the starting configuration before the motion is computed. "
        "Run the simulation from the sidebar to generate playback frames and optionally compare MPI."
    )
    preview, _, _, labels = two_clusters(n, seed, sep, vfrac, impact)
    st.plotly_chart(
        space_figure(preview, labels, mode="clusters", title="Initial star-cluster configuration"),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

st.divider()
st.subheader("Saved benchmark results")
if (RESULTS / "runs.csv").exists():
    with (RESULTS / "runs.csv").open(newline="") as f:
        stored = list(csv.DictReader(f))
    st.dataframe(stored[-50:][::-1], use_container_width=True)
else:
    st.write("No saved experiments yet.")
