"""Plot star-cluster collision frames as PNG or animated GIF, outside the timed simulation.

Requires a run made with --save-every K; can also plot final positions without frames.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def render(run_id, gif=False, fps=12, max_frames=120):
    data = np.load(RESULTS / f"final_{run_id}.npz")
    labels = data["label"]
    saved = RESULTS / f"frames_{run_id}.npz"
    frames = np.load(saved)["frames"] if saved.exists() else data["pos"][None, :, :]
    # Keep fixed coordinates to show the physical trajectory rather than rescaling each frame.
    selected = np.arange(len(frames))
    if len(selected) > max_frames:
        selected = np.unique(np.linspace(0, len(frames)-1, max_frames).astype(int))
    x = frames[selected, :, 0]; y = frames[selected, :, 1]
    min_x, max_x = np.percentile(x, [1, 99]); min_y, max_y = np.percentile(y, [1, 99])
    half = max((max_x - min_x), (max_y - min_y), 1.0) * 0.6
    cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
    fig, ax = plt.subplots(figsize=(7, 6))
    artists = [ax.scatter([], [], s=2, alpha=0.7, label=f"Cluster {i+1}") for i in (0,1)]
    ax.set(xlim=(cx-half, cx+half), ylim=(cy-half, cy+half), xlabel="x", ylabel="y", aspect="equal")
    ax.legend(loc="upper right")
    title = ax.set_title("Two Plummer spheres")

    def draw(index):
        frame = frames[int(selected[index])]
        for i, art in enumerate(artists):
            art.set_offsets(frame[labels == i, :2])
        title.set_text(f"N-body cluster collision | frame {int(selected[index])+1}/{len(frames)}")
        return [*artists, title]

    draw(len(selected)-1)
    image_path = RESULTS / f"collision_{run_id}.png"
    fig.savefig(image_path, dpi=140)
    print(f"Saved {image_path}")
    if gif:
        if len(frames) < 2:
            raise ValueError("GIF requires frames: rerun with --save-every 5 (or another positive K)")
        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, draw, frames=len(selected), interval=1000/fps, blit=False)
        gif_path = RESULTS / f"collision_{run_id}.gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"Saved {gif_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=120)
    args = ap.parse_args()
    if args.fps < 1 or args.max_frames < 2:
        ap.error("fps >= 1 and max-frames >= 2 required")
    render(args.run_id, args.gif, args.fps, args.max_frames)

if __name__ == "__main__":
    main()
