#!/usr/bin/env python3
"""Diagnostics for a PROTEUS SR dataset run.

The script compares 3D ground-truth microbubble positions against exported
2D LR B-mode frames. It reports likely label-noise sources for SR training:
out-of-plane bubbles, projected GT points with weak LR response, and tracks
that barely move across frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def _mat_struct_fields(obj):
    return getattr(obj, "_fieldnames", [])


def _as_1d(x):
    return np.asarray(x).reshape(-1)


def _load_settings(path: Path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _load_metadata(path: Path):
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["metadata"]
    out = {
        "x_lat_mm": _as_1d(md.x_lat_mm).astype(float),
        "z_ax_mm": _as_1d(md.z_ax_mm).astype(float),
        "image_size": tuple(int(v) for v in _as_1d(md.image_size)),
        "num_frames": int(md.num_frames),
        "dynamic_range": float(md.dynamic_range),
        "sigma_px": float(md.sigma_px),
    }
    if hasattr(md, "elevation_filter_mm"):
        out["elevation_filter_mm"] = float(md.elevation_filter_mm)
    else:
        out["elevation_filter_mm"] = None
    return out


def _load_frame(gt_file: Path):
    frame = sio.loadmat(gt_file, squeeze_me=True, struct_as_record=False)["Frame"]
    pulse = frame.Pulse1
    pts = np.asarray(pulse.Points, dtype=float)
    radii = _as_1d(pulse.Radius).astype(float)
    streams = _as_1d(pulse.StreamNumber).astype(int)
    velocity = np.asarray(pulse.Velocity, dtype=float)
    return pts, radii, streams, velocity


def _to_image_coords(points_m: np.ndarray, geometry):
    pts = points_m.T
    pts = pts - np.asarray(geometry.BoundingBox.Center, dtype=float).reshape(3, 1)
    pts = np.asarray(geometry.Rotation, dtype=float) @ pts
    pts = pts + np.asarray(geometry.Center, dtype=float).reshape(3, 1)
    pts = pts.T
    depth_mm = pts[:, 0] * 1e3
    lat_mm = pts[:, 1] * 1e3
    elev_mm = pts[:, 2] * 1e3
    return np.column_stack([lat_mm, depth_mm, elev_mm])


def _sample_lr_3x3(lr: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    out = np.full(rows.shape, np.nan, dtype=float)
    h, w = lr.shape
    valid = np.isfinite(rows) & np.isfinite(cols)
    valid &= (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    for idx in np.where(valid)[0]:
        r = int(round(rows[idx]))
        c = int(round(cols[idx]))
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        out[idx] = float(np.max(lr[r0:r1, c0:c1]))
    return out


def _pct(x, qs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(x, q)) for q in qs}


def diagnose(run_dir: Path, out_dir: Path, visibility_threshold: float):
    settings_path = run_dir / "settings.mat"
    gt_dir = run_dir / "ground_truth"
    metadata_path = run_dir / "dataset_sr" / "metadata.mat"
    blob_mat_dir = run_dir / "dataset_sr" / "mat" / "blob"

    settings = _load_settings(settings_path)
    geom = settings["Geometry"]
    md = _load_metadata(metadata_path)
    x_lat_mm = md["x_lat_mm"]
    z_ax_mm = md["z_ax_mm"]

    frame_files = sorted(gt_dir.glob("Frame_*.mat"))
    if not frame_files:
        raise FileNotFoundError(f"No Frame_*.mat files found in {gt_dir}")

    all_lat = []
    all_depth = []
    all_elev = []
    all_radius_um = []
    all_stream = []
    all_speed_mm_s = []
    all_lr = []
    all_in_grid = []

    positions_by_frame = []
    streams_by_frame = []

    for frame_idx, gt_file in enumerate(frame_files, start=1):
        pts, radii, streams, velocity = _load_frame(gt_file)
        img = _to_image_coords(pts, geom)
        lat, depth, elev = img[:, 0], img[:, 1], img[:, 2]

        positions_by_frame.append(img)
        streams_by_frame.append(streams)

        cols = np.interp(lat, x_lat_mm, np.arange(len(x_lat_mm)), left=np.nan, right=np.nan)
        rows = np.interp(depth, z_ax_mm, np.arange(len(z_ax_mm)), left=np.nan, right=np.nan)
        in_grid = np.isfinite(rows) & np.isfinite(cols)

        lr_vals = np.full(lat.shape, np.nan, dtype=float)
        lr_candidates = []
        for width in range(1, 7):
            candidate = blob_mat_dir / f"frame_{frame_idx:0{width}d}.mat"
            if candidate.exists():
                lr_candidates.append(candidate)
        lr_file = lr_candidates[0] if lr_candidates else blob_mat_dir / f"frame_{frame_idx:03d}.mat"
        if lr_file.exists():
            lr_frame = sio.loadmat(lr_file, squeeze_me=True, struct_as_record=False)["lr_frame"]
            lr_vals = _sample_lr_3x3(np.asarray(lr_frame, dtype=float), rows, cols)

        all_lat.append(lat)
        all_depth.append(depth)
        all_elev.append(elev)
        all_radius_um.append(radii * 1e6)
        all_stream.append(streams)
        all_speed_mm_s.append(np.linalg.norm(velocity, axis=1) * 1e3)
        all_lr.append(lr_vals)
        all_in_grid.append(in_grid)

    lat = np.concatenate(all_lat)
    depth = np.concatenate(all_depth)
    elev = np.concatenate(all_elev)
    radius_um = np.concatenate(all_radius_um)
    speed_mm_s = np.concatenate(all_speed_mm_s)
    lr_vals = np.concatenate(all_lr)
    in_grid = np.concatenate(all_in_grid)

    visible = np.isfinite(lr_vals) & (lr_vals >= visibility_threshold)
    weak = np.isfinite(lr_vals) & (lr_vals < visibility_threshold)
    if md["elevation_filter_mm"] is None or np.isinf(md["elevation_filter_mm"]):
        export_label_mask = np.ones(elev.shape, dtype=bool)
    else:
        export_label_mask = np.abs(elev) <= md["elevation_filter_mm"]
    export_finite = export_label_mask & np.isfinite(lr_vals)

    displacements_2d = []
    displacements_3d = []
    same_stream = []
    for prev_pos, pos, prev_stream, stream in zip(
        positions_by_frame[:-1],
        positions_by_frame[1:],
        streams_by_frame[:-1],
        streams_by_frame[1:],
    ):
        n = min(len(prev_pos), len(pos))
        dsame = prev_stream[:n] == stream[:n]
        delta = pos[:n] - prev_pos[:n]
        displacements_2d.append(np.linalg.norm(delta[:, :2], axis=1))
        displacements_3d.append(np.linalg.norm(delta, axis=1))
        same_stream.append(dsame)

    disp2d = np.concatenate(displacements_2d)
    disp3d = np.concatenate(displacements_3d)
    same_stream = np.concatenate(same_stream)
    disp2d_same = disp2d[same_stream]
    disp3d_same = disp3d[same_stream]
    static_2d = same_stream & (disp2d < 0.01)
    static_3d = same_stream & (disp3d < 0.01)

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_dir": str(run_dir),
        "frames": len(frame_files),
        "points_total": int(lat.size),
        "settings": {
            "transducer": str(settings["Transducer"].Type),
            "center_frequency_mhz": float(settings["Transducer"].CenterFrequency / 1e6),
            "pressure_kpa": float(settings["Transmit"].AcousticPressure / 1e3),
            "mechanical_index": float(settings["Transmit"].MechanicalIndex),
            "frame_rate_hz": float(settings["Acquisition"].FrameRate),
            "microbubbles": int(settings["Microbubble"].Number),
        },
        "image_grid": {
            "lat_mm_minmax": [float(np.nanmin(x_lat_mm)), float(np.nanmax(x_lat_mm))],
            "depth_mm_minmax": [float(np.nanmin(z_ax_mm)), float(np.nanmax(z_ax_mm))],
            "size": list(md["image_size"]),
        },
        "gt_distribution": {
            "lat_mm_percentiles": _pct(lat, [0, 5, 50, 95, 100]),
            "depth_mm_percentiles": _pct(depth, [0, 5, 50, 95, 100]),
            "abs_elev_mm_percentiles": _pct(np.abs(elev), [0, 50, 75, 90, 95, 99, 100]),
            "radius_um_percentiles": _pct(radius_um, [0, 50, 75, 90, 95, 99, 100]),
            "speed_mm_s_percentiles_unscaled_field": _pct(speed_mm_s, [0, 5, 50, 95, 100]),
        },
        "label_visibility_proxy": {
            "threshold_lr_0_to_1": visibility_threshold,
            "raw_gt_note": "All GT points before any export-time elevation filtering.",
            "in_grid_fraction": float(np.mean(in_grid)),
            "sampled_points": int(np.sum(np.isfinite(lr_vals))),
            "visible_fraction": float(np.mean(visible[np.isfinite(lr_vals)])),
            "weak_fraction": float(np.mean(weak[np.isfinite(lr_vals)])),
            "lr_value_percentiles": _pct(lr_vals, [0, 25, 50, 75, 90, 95, 99, 100]),
            "weak_abs_elev_mm_percentiles": _pct(np.abs(elev[weak]), [50, 75, 90, 95, 99]),
            "visible_abs_elev_mm_percentiles": _pct(np.abs(elev[visible]), [50, 75, 90, 95, 99]),
        },
        "exported_label_visibility_proxy": {
            "elevation_filter_mm_from_metadata": md["elevation_filter_mm"],
            "exported_label_fraction_of_raw_gt": float(np.mean(export_label_mask)),
            "sampled_exported_points": int(np.sum(export_finite)),
            "visible_fraction_exported": (
                float(np.mean(visible[export_finite])) if np.any(export_finite) else None
            ),
            "weak_fraction_exported": (
                float(np.mean(weak[export_finite])) if np.any(export_finite) else None
            ),
            "exported_lr_value_percentiles": _pct(lr_vals[export_finite], [0, 25, 50, 75, 90, 95, 99, 100]),
        },
        "motion": {
            "same_stream_pairs": int(np.sum(same_stream)),
            "reseed_pairs": int(np.sum(~same_stream)),
            "disp2d_mm_percentiles_same_stream": _pct(disp2d_same, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
            "disp3d_mm_percentiles_same_stream": _pct(disp3d_same, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
            "static_2d_pairs_lt_0p01mm_fraction": float(np.mean(static_2d[same_stream])),
            "static_3d_pairs_lt_0p01mm_fraction": float(np.mean(static_3d[same_stream])),
        },
        "elevation_filter_candidates": {},
    }

    finite_lr = np.isfinite(lr_vals)
    for threshold_mm in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        keep = np.abs(elev) <= threshold_mm
        keep_finite = keep & finite_lr
        summary["elevation_filter_candidates"][f"abs_elev_le_{threshold_mm:g}mm"] = {
            "kept_fraction_all_gt": float(np.mean(keep)),
            "visible_fraction_kept": float(np.mean(visible[keep_finite])) if np.any(keep_finite) else None,
            "weak_fraction_kept": float(np.mean(weak[keep_finite])) if np.any(keep_finite) else None,
        }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    axs[0].hist(depth, bins=60, color="steelblue")
    axs[0].set_title("GT depth distribution")
    axs[0].set_xlabel("depth [mm]")
    axs[1].hist(np.abs(elev), bins=60, color="darkorange")
    axs[1].set_title("GT |elevation| distribution")
    axs[1].set_xlabel("|elevation| [mm]")
    axs[2].hist(radius_um, bins=60, color="seagreen")
    axs[2].set_title("Sampled MB radius")
    axs[2].set_xlabel("radius [um]")
    axs[3].hist(lr_vals[np.isfinite(lr_vals)], bins=60, color="slategray")
    axs[3].axvline(visibility_threshold, color="red", linestyle="--")
    axs[3].set_title("LR intensity at GT projection")
    axs[3].set_xlabel("3x3 max LR value")
    axs[4].hist(disp2d_same, bins=60, color="purple")
    axs[4].set_title("Same-stream 2D displacement")
    axs[4].set_xlabel("mm/frame")
    axs[5].scatter(np.abs(elev[finite_lr]), lr_vals[finite_lr], s=2, alpha=0.25)
    axs[5].axhline(visibility_threshold, color="red", linestyle="--")
    axs[5].set_title("Visibility proxy vs |elevation|")
    axs[5].set_xlabel("|elevation| [mm]")
    axs[5].set_ylabel("3x3 max LR")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic_plots.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote diagnostics to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Run folder, e.g. run_20260509_101335")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output diagnostics folder. Defaults to RUN_DIR/diagnostics.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.25,
        help="LR frame intensity threshold in [0,1] used as a rough visibility proxy.",
    )
    args = parser.parse_args()
    out = args.out if args.out is not None else args.run_dir / "diagnostics"
    diagnose(args.run_dir, out, args.visibility_threshold)


if __name__ == "__main__":
    main()
