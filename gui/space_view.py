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
PLANET_MARKERS = (30, 10, 12, 13, 15)


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
    vel[0] = -np.sum(mass[1:, None] * vel[1:], axis=0) / mass[0]
    pos[0] = -np.sum(mass[1:, None] * pos[1:], axis=0) / mass[0]

    def acceleration(position):
        d = position[None, :, :] - position[:, None, :]
        d2 = np.sum(d * d, axis=2) + 0.025**2
        return np.sum(mass[None, :, None] * d * d2[:, :, None] ** (-1.5), axis=1)

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
        half = max(3.7, float(np.max(np.abs(frames[..., :2]))) * 1.12)
        return [-half, half]
    xy = frames[..., :2].reshape(-1, 2)
    qlo, qhi = np.percentile(xy, [2, 98], axis=0)
    half = max(float(np.max(qhi - qlo)) * 0.64, 4.0)
    center = (qlo + qhi) * 0.5
    return [float(center[0] - half), float(center[0] + half)], [float(center[1] - half), float(center[1] + half)]


def _trails_for_group(history, ids):
    """Return a single polyline trace encoding trailing paths for many bodies."""
    if history.shape[0] < 2 or len(ids) == 0:
        return np.array([]), np.array([]), np.array([])
    seg_x, seg_y, seg_z = [], [], []
    hist = history[:, ids, :]
    for j in range(hist.shape[1]):
        seg_x.extend(hist[:, j, 0].tolist() + [None])
        seg_y.extend(hist[:, j, 1].tolist() + [None])
        seg_z.extend(hist[:, j, 2].tolist() + [None])
    return np.array(seg_x, dtype=object), np.array(seg_y, dtype=object), np.array(seg_z, dtype=object)


def _make_background(seed=715):
    rng = np.random.default_rng(seed)
    stars_far = rng.uniform(-18, 18, size=(300, 3))
    stars_mid = rng.normal(0.0, 9.0, size=(130, 3))
    stars_mid[:, 2] *= 0.45
    nebula_a = rng.normal(loc=[-7.5, 6.5, 0.0], scale=[2.3, 1.7, 1.2], size=(45, 3))
    nebula_b = rng.normal(loc=[8.0, -6.0, 0.0], scale=[2.6, 1.9, 1.3], size=(45, 3))
    return stars_far, stars_mid, nebula_a, nebula_b


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

    chosen = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int))
    samples = frames[chosen]
    names = ("Cluster A", "Cluster B") if mode == "clusters" else PLANET_NAMES
    colors = CLUSTER_COLORS if mode == "clusters" else PLANET_COLORS
    groups = [np.flatnonzero(labels == k) for k in range(len(names))]
    stars_far, stars_mid, nebula_a, nebula_b = _make_background()

    traces = [
        go.Scatter3d(
            x=stars_far[:, 0], y=stars_far[:, 1], z=stars_far[:, 2],
            mode="markers", marker=dict(size=1, color="#BFD5FF", opacity=0.26),
            showlegend=False, hoverinfo="skip", name="Background stars"
        ),
        go.Scatter3d(
            x=stars_mid[:, 0], y=stars_mid[:, 1], z=stars_mid[:, 2],
            mode="markers", marker=dict(size=np.linspace(1.3, 3.1, len(stars_mid)), color="#E8F1FF", opacity=0.38),
            showlegend=False, hoverinfo="skip", name="Background glow"
        ),
        go.Scatter3d(
            x=nebula_a[:, 0], y=nebula_a[:, 1], z=nebula_a[:, 2],
            mode="markers", marker=dict(size=13, color="rgba(100, 170, 255, 0.045)"),
            showlegend=False, hoverinfo="skip", name="Decorative blue nebula"
        ),
        go.Scatter3d(
            x=nebula_b[:, 0], y=nebula_b[:, 1], z=nebula_b[:, 2],
            mode="markers", marker=dict(size=13, color="rgba(255, 152, 104, 0.05)"),
            showlegend=False, hoverinfo="skip", name="Decorative orange nebula"
        ),
    ]

    moving_indices = []
    trail_indices = []
    centroid_indices = []

    if mode == "clusters":
        trace_names = []
        for k, ids in enumerate(groups):
            p = samples[0, ids]
            core_ids = ids.astype(int)
            custom = np.column_stack([core_ids, np.full(len(ids), k)])
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=9, color=colors[k], opacity=0.08),
                showlegend=False, hoverinfo="skip", name=f"{names[k]} glow"
            ))
            moving_indices.append(len(traces) - 1)
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=4.6, color=colors[k], opacity=0.98,
                            line=dict(color="rgba(255,255,255,0.32)", width=0.5)),
                name=names[k],
                customdata=custom,
                hovertemplate=(
                    f"<b>{names[k]} star</b><br>"
                    "Body index: %{customdata[0]}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                )
            ))
            moving_indices.append(len(traces) - 1)
            hx, hy, hz = _trails_for_group(samples[:min(6, len(samples))], ids)
            traces.append(go.Scatter3d(
                x=hx, y=hy, z=hz, mode="lines",
                line=dict(color=colors[k], width=1.2), opacity=0.16,
                showlegend=False, hoverinfo="skip", name=f"{names[k]} trails"
            ))
            trail_indices.append(len(traces) - 1)
            centroid = p.mean(axis=0)
            traces.append(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]], mode="markers",
                marker=dict(size=18, color=colors[k], opacity=0.16, symbol="diamond"),
                showlegend=False,
                hovertemplate=f"<b>{names[k]} center</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
                name=f"{names[k]} center"
            ))
            centroid_indices.append(len(traces) - 1)
            trace_names.extend([names[k]])
    else:
        for k, ids in enumerate(groups):
            p = samples[0, ids]
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=max(PLANET_MARKERS[k] + 10, 14), color=colors[k], opacity=0.12),
                showlegend=False, hoverinfo="skip", name=f"{names[k]} glow"
            ))
            moving_indices.append(len(traces) - 1)
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=PLANET_MARKERS[k], color=colors[k], opacity=0.98,
                            line=dict(color="rgba(255,255,255,0.25)", width=1.0)),
                name=names[k],
                hovertemplate=f"<b>{names[k]}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
            ))
            moving_indices.append(len(traces) - 1)
        for k in range(1, 5):
            tr = samples[:, k]
            traces.append(go.Scatter3d(
                x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode="lines",
                line=dict(color=colors[k], width=2), opacity=0.42,
                showlegend=False, hoverinfo="skip", name=f"{names[k]} orbit"
            ))

    anim_frames = []
    for idx, frame_pos in enumerate(samples):
        frame_data = []
        frame_traces = []
        if mode == "clusters":
            start = max(0, idx - 5)
            history = samples[start: idx + 1]
            for k, ids in enumerate(groups):
                p = frame_pos[ids]
                frame_data.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]))
                frame_traces.append(moving_indices[2 * k])
                frame_data.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]))
                frame_traces.append(moving_indices[2 * k + 1])
                hx, hy, hz = _trails_for_group(history, ids)
                frame_data.append(go.Scatter3d(x=hx, y=hy, z=hz))
                frame_traces.append(trail_indices[k])
                centroid = p.mean(axis=0)
                frame_data.append(go.Scatter3d(x=[centroid[0]], y=[centroid[1]], z=[centroid[2]]))
                frame_traces.append(centroid_indices[k])
        else:
            for k, ids in enumerate(groups):
                p = frame_pos[ids]
                frame_data.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]))
                frame_traces.append(moving_indices[2 * k])
                frame_data.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]))
                frame_traces.append(moving_indices[2 * k + 1])
        anim_frames.append(go.Frame(name=str(idx), traces=frame_traces, data=frame_data))

    if mode == "planets":
        xr = yr = _range_from_frames(samples, mode)
        zr = [-2, 2]
    else:
        xr, yr = _range_from_frames(samples, mode)
        zr = [-max(4, (xr[1] - xr[0]) * 0.43), max(4, (xr[1] - xr[0]) * 0.43)]

    scene = dict(
        bgcolor="#020611",
        aspectmode="cube",
        xaxis=dict(range=xr, visible=False, showbackground=False),
        yaxis=dict(range=yr, visible=False, showbackground=False),
        zaxis=dict(range=zr, visible=False, showbackground=False),
        camera=dict(eye=dict(x=1.42, y=1.40, z=0.88)),
    )

    fig = go.Figure(data=traces, frames=anim_frames)
    fig.update_layout(
        title=dict(
            text=title or ("Real N-Body Star-Cluster Collision" if mode == "clusters" else "Optional Planetary Demo"),
            font=dict(color="#E6EFFF", size=22)
        ),
        paper_bgcolor="#020611",
        plot_bgcolor="#020611",
        font=dict(color="#D7E4FF"),
        margin=dict(l=0, r=0, t=52, b=0),
        height=640,
        scene=scene,
        legend=dict(
            orientation="h", x=0.03, y=1.02,
            bgcolor="rgba(4,9,22,0.55)",
            font=dict(color="#E2E8F0")
        ),
        uirevision="space",
    )

    if len(samples) > 1:
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.02, y=0.055,
                direction="right", bgcolor="#24324D", bordercolor="#55637F",
                font=dict(color="#FFFFFF"),
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None,
                        dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                             transition=dict(duration=0), mode="immediate")]),
                    dict(label="❚❚ Pause", method="animate", args=[[None],
                        dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                ]
            )],
            sliders=[dict(
                active=0, x=0.19, y=0.08, len=0.78,
                currentvalue=dict(prefix="Frame ", font=dict(color="#E2E8F0")),
                bgcolor="rgba(12,18,34,0.52)",
                steps=[dict(label=str(int(chosen[i])), method="animate", args=[[str(i)],
                    dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                    for i in range(len(chosen))]
            )]
        )
    return fig
