#!/usr/bin/env python3
"""Render presentation-style SR visualization variants from exported data.

This works from dataset_sr outputs, so it does not rerun RF simulation or DAS.
It is intended for quick local comparison of display-only cleanup choices:
contrast clipping, thresholding, gamma, and GT coordinate overlays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter


def _as_1d(value):
    return np.asarray(value).reshape(-1)


def load_metadata(run_dir: Path):
    md = sio.loadmat(
        run_dir / "dataset_sr" / "metadata.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["metadata"]
    return {
        "x_lat_mm": _as_1d(md.x_lat_mm).astype(float),
        "z_ax_mm": _as_1d(md.z_ax_mm).astype(float),
        "num_frames": int(md.num_frames),
        "elevation_filter_mm": float(md.elevation_filter_mm)
        if hasattr(md, "elevation_filter_mm")
        else np.inf,
    }


def frame_tag(frame_idx: int, num_frames: int) -> str:
    return f"{frame_idx:0{len(str(num_frames))}d}"


def load_frame(run_dir: Path, frame_idx: int, num_frames: int):
    tag = frame_tag(frame_idx, num_frames)
    lr_path = run_dir / "dataset_sr" / "mat" / "blob" / f"frame_{tag}.mat"
    coord_path = run_dir / "dataset_sr" / "coordinates" / f"frame_{tag}.mat"
    lr = sio.loadmat(lr_path, squeeze_me=True, struct_as_record=False)["lr_frame"]
    coords = sio.loadmat(coord_path, squeeze_me=True, struct_as_record=False)
    gt_mm = np.asarray(coords["gt_coords_mm"], dtype=float).reshape(-1, 2)
    gt_elev_mm = np.asarray(coords.get("gt_elev_mm", np.zeros(len(gt_mm))), dtype=float).reshape(-1)
    return np.asarray(lr, dtype=float), gt_mm, gt_elev_mm


def enhance(lr: np.ndarray, mode: str) -> np.ndarray:
    img = np.clip(lr, 0, 1)
    if mode == "raw":
        return img
    if mode == "clip_q99_gamma":
        hi = np.percentile(img, 99.2)
        img = np.clip(img / max(hi, 1e-6), 0, 1)
        return img**1.35
    if mode == "threshold_gamma":
        lo = 0.18
        img = np.clip((img - lo) / (1 - lo), 0, 1)
        return img**1.55
    if mode == "bg_subtract":
        bg = gaussian_filter(img, sigma=8)
        img = np.clip(img - 0.75 * bg, 0, None)
        hi = np.percentile(img, 99.5)
        img = np.clip(img / max(hi, 1e-6), 0, 1)
        return img**1.25
    if mode == "bright_only":
        lo = np.percentile(img, 92)
        img = np.clip((img - lo) / max(1 - lo, 1e-6), 0, 1)
        return img**1.2
    raise ValueError(f"Unknown mode: {mode}")


def plot_single(ax, img, x_lat_mm, z_ax_mm, title, gt_mm=None, gt_elev_mm=None, elev_filter=None):
    extent = [x_lat_mm[0], x_lat_mm[-1], z_ax_mm[-1], z_ax_mm[0]]
    ax.imshow(img, cmap="gray", extent=extent, aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Width [mm]", fontsize=8)
    ax.set_ylabel("Depth [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    if gt_mm is not None and len(gt_mm):
        keep = np.ones(len(gt_mm), dtype=bool)
        if gt_elev_mm is not None and elev_filter is not None and np.isfinite(elev_filter):
            keep = np.abs(gt_elev_mm) <= elev_filter
        pts = gt_mm[keep]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            "*",
            color="#00a2ff",
            markersize=4,
            markeredgewidth=0.6,
            linestyle="None",
        )


def render_frame(run_dir: Path, out_dir: Path, frame_idx: int, modes: list[str], overlay: bool):
    md = load_metadata(run_dir)
    lr, gt_mm, gt_elev_mm = load_frame(run_dir, frame_idx, md["num_frames"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 5.2), constrained_layout=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        plot_single(
            ax,
            enhance(lr, mode),
            md["x_lat_mm"],
            md["z_ax_mm"],
            mode.replace("_", " "),
            gt_mm if overlay else None,
            gt_elev_mm if overlay else None,
            md["elevation_filter_mm"],
        )
    fig.suptitle(f"{run_dir.name} - frame {frame_idx:02d}", fontsize=12)
    fig.savefig(out_dir / f"frame_{frame_idx:02d}_variants.png", dpi=180)
    plt.close(fig)

    # Save the recommended display candidate separately for quick inspection.
    fig2, ax2 = plt.subplots(figsize=(7.2, 7.8), constrained_layout=True)
    plot_single(
        ax2,
        enhance(lr, "bg_subtract"),
        md["x_lat_mm"],
        md["z_ax_mm"],
        f"{run_dir.name} - frame {frame_idx:02d}",
        gt_mm if overlay else None,
        gt_elev_mm if overlay else None,
        md["elevation_filter_mm"],
    )
    fig2.savefig(out_dir / f"frame_{frame_idx:02d}_recommended.png", dpi=180)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("run_20260523_110555"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 25, 50])
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir / "visualizations" / "sr_variants"
    modes = ["raw", "clip_q99_gamma", "threshold_gamma", "bg_subtract", "bright_only"]
    for frame_idx in args.frames:
        render_frame(args.run_dir, out_dir, frame_idx, modes, overlay=not args.no_overlay)
    print(f"Wrote SR visualization variants to {out_dir}")


if __name__ == "__main__":
    main()
