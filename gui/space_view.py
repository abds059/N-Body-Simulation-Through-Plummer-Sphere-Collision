"""Interactive Plotly scenes for the Streamlit demo.

Rendering is outside all simulation timers. Positions come only from saved
simulation frames or from the same direct-gravity numerical kernel in the
optional toy planetary system.
"""

import numpy as np
import plotly.graph_objects as go

CLUSTER_COLORS = ("#6DE6FF", "#FF9E68")
PLANET_NAMES = ("Sun", "Aqua", "Ember", "Verdant", "Violet")
PLANET_COLORS = ("#FFDB75", "#69BEFF", "#FA917B", "#8EDE9B", "#AFA1FF")
PLANET_MARKERS = (25, 9, 11, 12, 14)


def planetary_frames(steps=2300, dt=0.015, save_every=15):
    """Toy 5-body orbit example; not real Solar System data or MPI benchmarks.

    Uses G=1, Plummer-style force softening, and leapfrog kick-drift-kick,
    consistent with the main solver's numerical method.
    """
    if steps < 1 or dt <= 0 or save_every < 1:
        raise ValueError("steps >= 1, dt > 0, save_every >= 1 required")
    radius = np.array([1.0, 1.55, 2.25, 3.05])
    phases = np.array([0.2, 2.7, 4.25, 5.0])
    pos = np.zeros((5, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    mass = np.array([0.9999, 2e-5, 2.5e-5, 3e-5, 2.5e-5], dtype=np.float64)
    for i, (r, theta) in enumerate(zip(radius, phases), 1):
        pos[i, 0] = r * np.cos(theta)
        pos[i, 1] = r * np.sin(theta)
        v = np.sqrt(mass[0] / r)
        vel[i] = [-v * np.sin(theta), v * np.cos(theta), 0.0]
    # The central mass reacts to the planets, instead of being pinned down.
    vel[0] = -np.sum(mass[1:, None] * vel[1:], axis=0) / mass[0]
    pos[0] = -np.sum(mass[1:, None] * pos[1:], axis=0) / mass[0]

    def acceleration(position):
        d = position[None, :, :] - position[:, None, :]
        d2 = np.sum(d*d, axis=2) + 0.025**2
        return np.sum(mass[None, :, None] * d * d2[:, :, None]**(-1.5), axis=1)

    frames = [pos.copy()]
    a = acceleration(pos)
    for step in range(1, steps + 1):
        vel += 0.5 * dt * a
        pos += dt * vel
        a = acceleration(pos)
        vel += 0.5 * dt * a
        if step % save_every == 0:
            frames.append(pos.copy())
    return np.array(frames), np.arange(5, dtype=np.int8)


def _range_from_frames(frames, mode):
    if mode == "planets":
        half = max(3.7, float(np.max(np.abs(frames[..., :2])))*1.12)
        return [-half, half]
    xy = frames[..., :2].reshape(-1, 2)
    # Rare remote stars can be far beyond the clusters. Keep their central
    # interaction visible by using a robust view window.
    qlo, qhi = np.percentile(xy, [2, 98], axis=0)
    half = max(float(np.max(qhi-qlo))*0.64, 4.0)
    center = (qlo+qhi)*0.5
    return [float(center[0]-half), float(center[0]+half)], [float(center[1]-half), float(center[1]+half)]


def space_figure(raw_frames, labels, mode="clusters", max_frames=95, title=None):
    """Create a draggable 3D Plotly scene with play/pause and frame slider."""
    frames = np.asarray(raw_frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, :, :]
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError("Expected frame coordinates of shape (frames, N, 3)")
    labels = np.asarray(labels)
    if frames.shape[1] != len(labels):
        raise ValueError("Labels must match the number of bodies")
    if mode not in {"clusters", "planets"}:
        raise ValueError("mode must be 'clusters' or 'planets'")
    chosen = np.unique(np.linspace(0, len(frames)-1, min(max_frames, len(frames)), dtype=int))
    samples = frames[chosen]
    names = ("Cluster A", "Cluster B") if mode == "clusters" else PLANET_NAMES
    colors = CLUSTER_COLORS if mode == "clusters" else PLANET_COLORS
    groups = [np.flatnonzero(labels == k) for k in range(len(names))]
    traces = []
    rng = np.random.default_rng(715)
    # Background stars are visual decoration, NOT simulated bodies.
    background = rng.uniform(-18, 18, size=(165, 3))
    traces.append(go.Scatter3d(
        x=background[:, 0], y=background[:, 1], z=background[:, 2],
        mode="markers", marker=dict(size=1, color="#6E7892", opacity=0.18),
        showlegend=False, hoverinfo="skip", name="Decorative background"))
    for k, ids in enumerate(groups):
        p = samples[0, ids]
        traces.append(go.Scatter3d(
            x=p[:,0], y=p[:,1], z=p[:,2], mode="markers",
            marker=dict(size=(PLANET_MARKERS[k] if mode == "planets" else 4),
                        color=colors[k], opacity=0.94),
            name=names[k], hovertemplate=f"{names[k]}<extra></extra>"))

    # In planetary mode, orbit trails reflect the simulated positions.
    # They're static over the whole trajectory for context as bodies move.
    if mode == "planets":
        for k in range(1, 5):
            tr = samples[:,k]
            traces.append(go.Scatter3d(
                x=tr[:,0], y=tr[:,1], z=tr[:,2], mode="lines",
                line=dict(color=colors[k], width=2),
                opacity=0.35, showlegend=False, hoverinfo="skip"))

    anim_frames = []
    moving_indices = list(range(1, 1+len(groups)))
    for idx, frame_pos in enumerate(samples):
        anim_frames.append(go.Frame(
            name=str(idx), traces=moving_indices,
            data=[go.Scatter3d(x=frame_pos[ids,0], y=frame_pos[ids,1],
                               z=frame_pos[ids,2]) for ids in groups]))

    if mode == "planets":
        xr = yr = _range_from_frames(samples, mode)
        zr = [-2, 2]
    else:
        xr, yr = _range_from_frames(samples, mode)
        zr = [-max(4, (xr[1]-xr[0])*0.43), max(4, (xr[1]-xr[0])*0.43)]
    scene = dict(
        bgcolor="#050915", aspectmode="cube", xaxis=dict(range=xr, visible=False),
        yaxis=dict(range=yr, visible=False), zaxis=dict(range=zr, visible=False),
        camera=dict(eye=dict(x=1.42, y=1.40, z=0.88)))
    fig = go.Figure(data=traces, frames=anim_frames)
    fig.update_layout(
        title=dict(text=title or ("3D Star-Cluster Collision" if mode=="clusters" else "Toy Planetary Orbits"),
                   font=dict(color="#E6EFFF", size=20)),
        paper_bgcolor="#050915", plot_bgcolor="#050915", font=dict(color="#CBD5E1"),
        margin=dict(l=0,r=0,t=48,b=0), height=610, scene=scene,
        legend=dict(orientation="h", x=0.04, y=1.0, font=dict(color="#E2E8F0")),
        uirevision="space",
    )
    if len(samples) > 1:
        fig.update_layout(
            updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                direction="right", bgcolor="#253048", font=dict(color="#FFFFFF"),
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None,
                        dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                             transition=dict(duration=0), mode="immediate")]),
                    dict(label="❚❚ Pause", method="animate", args=[[None],
                        dict(frame=dict(duration=0, redraw=True), mode="immediate")])])],
            sliders=[dict(active=0, x=0.19, y=0.08, len=0.78,
                currentvalue=dict(prefix="Frame ", font=dict(color="#E2E8F0")),
                steps=[dict(label=str(int(chosen[i])), method="animate", args=[[str(i)],
                    dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                    for i in range(len(chosen))])]
        )
    return fig
