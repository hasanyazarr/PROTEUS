#!/usr/bin/env python3
"""Create an advisor-facing v7 versus v8_fat simulation diagnostic figure."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[5]
RUNS = {
    "v7": ROOT / "runs" / "simulation" / "run_20260708_043218_merged",
    "v8_fat": ROOT / "runs" / "simulation" / "run_20260712_015537",
}
OUT = ROOT / "outputs" / "figures" / "v7_vs_v8_fat_diagnostic.png"


def frame_number(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def frame_files(run: Path, folder: str) -> list[Path]:
    return sorted((run / folder).glob("Frame_*.mat"), key=frame_number)


def rf(path: Path) -> tuple[np.ndarray, float]:
    data = loadmat(path, variable_names=["RF", "dt"], simplify_cells=True)
    return np.asarray(data["RF"], dtype=np.float64), float(data["dt"])


def moving_rms_fraction(files: list[Path]) -> tuple[float, float]:
    """Return median relative RMS difference and correlation to frame 1."""
    base, _ = rf(files[0])
    base_rms = np.sqrt(np.mean(base**2))
    differences, correlations = [], []
    for path in files[1:]:
        current, _ = rf(path)
        differences.append(np.sqrt(np.mean((current - base) ** 2)) / base_rms)
        correlations.append(np.corrcoef(base.ravel(), current.ravel())[0, 1])
    return float(np.median(differences)), float(np.median(correlations))


def label_counts(run: Path) -> Counter:
    counts: Counter = Counter()
    for path in sorted((run / "dataset_sr" / "coordinates").glob("frame_*.mat")):
        data = loadmat(path, simplify_cells=True)["DroppedLabelCountsByReason"]
        counts.update({key: int(value) for key, value in data.items()})
    return counts


def main() -> None:
    old_files = frame_files(RUNS["v7"], "RF_data")
    fat_files = frame_files(RUNS["v8_fat"], "RF_data")
    old_rf, old_dt = rf(old_files[0])
    fat_rf, fat_dt = rf(fat_files[0])

    # A representative central channel and smoothed absolute envelope make the
    # acoustic trace legible at advisor-slide scale.
    channel = old_rf.shape[0] // 2
    def envelope(signal: np.ndarray) -> np.ndarray:
        return np.convolve(np.abs(signal), np.ones(64) / 64, mode="same")

    old_env = envelope(old_rf[channel])
    fat_env = envelope(fat_rf[channel])
    old_time = np.arange(old_rf.shape[1]) * old_dt * 1e6
    fat_time = np.arange(fat_rf.shape[1]) * fat_dt * 1e6
    old_delta, old_corr = moving_rms_fraction(old_files)
    fat_delta, fat_corr = moving_rms_fraction(fat_files)

    labels = label_counts(RUNS["v8_fat"])
    total_labels = sum(labels.values())
    valid_rate = labels["valid"] / total_labels * 100

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(15, 5.8))
    axes = fig.subplot_mosaic([["params", "signal", "labels"]], width_ratios=[1.0, 1.5, 1.0])
    fig.suptitle("PROTEUS diagnostic: v7 general tissue vs v8_fat", fontsize=16, fontweight="bold", y=0.975)

    # Panel A: only parameters that actually changed in settings.mat.
    ax = axes["params"]
    names = ["Sound speed (m/s)", "Density (kg/m³)", "Inhomogeneity", "Atten. A", "Atten. B", "B/A"]
    old = np.array([1540, 1000, 0.020, 0.75, 1.50, 6.0])
    fat = np.array([1450, 950, 0.005, 0.60, 1.01, 10.0])
    y = np.arange(len(names))
    # Relative indexing avoids false comparisons between differently scaled units.
    ax.barh(y - 0.18, np.full(len(names), 100), height=0.34, color="#4C78A8", label="v7")
    ax.barh(y + 0.18, fat / old * 100, height=0.34, color="#F58518", label="v8_fat")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 185)
    ax.set_xlabel("Value indexed to v7 = 100")
    ax.set_title("Actual run-setting changes", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.98, 0.97, "Blue: v7\nOrange: v8_fat", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5)
    for i, (a, b) in enumerate(zip(old, fat)):
        ax.text(max(100, b / a * 100) + 3, i + 0.18, f"{a:g} → {b:g}", va="center", fontsize=8)
    ax.text(0.0, 0.02, "Same: 200 bubbles, 500 Hz, 3DG solver.\nFrames: 500 → 50.", transform=ax.transAxes, fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9})

    # Panel B: RF envelope plus the temporal-stability warning.
    ax = axes["signal"]
    ax.plot(old_time, old_env, color="#4C78A8", lw=1.2, label="v7 (1540 m/s)")
    ax.plot(fat_time, fat_env, color="#F58518", lw=1.2, label="v8_fat (1450 m/s)")
    ax.set_xlim(0, max(old_time[-1], fat_time[-1]))
    ax.set_xlabel("Receive time (µs)")
    ax.set_ylabel("Smoothed |RF|, central channel")
    ax.set_title("Representative RF signal", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02, 0.96,
        f"Trace duration: {old_time[-1]:.1f} → {fat_time[-1]:.1f} µs\n"
        f"Median frame-to-frame change: {old_delta * 100:.4f}% / {fat_delta * 100:.4f}%\n"
        f"RF correlation to frame 1: {old_corr:.8f} / {fat_corr:.8f}",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.94},
    )
    ax.annotate("Near-static temporal RF\nrelative to moving GT", xy=(73, np.max(fat_env) * 0.52), xytext=(45, np.max(fat_env) * 0.78),
                arrowprops={"arrowstyle": "->", "color": "0.25"}, fontsize=9, color="0.15")

    # Panel C: valid labels versus exclusion reasons.
    ax = axes["labels"]
    order = ["valid", "weak_response", "out_of_plane", "out_of_fov"]
    colors = ["#54A24B", "#ECA82C", "#E45756", "#B0B0B0"]
    left = 0
    for reason, color in zip(order, colors):
        count = labels[reason]
        pct = count / total_labels * 100
        ax.barh([0], [pct], left=left, color=color, height=0.48, label=reason.replace("_", " "))
        if pct >= 5:
            label = f"{pct:.1f}%" if reason != "out_of_plane" else f"out of plane\n{pct:.1f}%"
            ax.text(left + pct / 2, 0, label, ha="center", va="center", fontsize=9 if reason != "out_of_plane" else 10, fontweight="bold")
        left += pct
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of 10,000 simulated bubble instances")
    ax.set_title("v8_fat exported-label retention", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.0, 0.95, f"{valid_rate:.2f}% valid = {labels['valid']:,} labels\n23.5 valid labels / frame",
            transform=ax.transAxes, va="top", fontsize=12, fontweight="bold")
    ax.text(0.0, 0.08, "Dominant exclusion: ±1 mm elevation filter\n(not lateral/axial FOV).", transform=ax.transAxes, fontsize=9)

    fig.text(0.01, 0.012, "Sources: each run's settings.mat, RF_data, ground_truth, and v8_fat dataset_sr/coordinates. "
             "Interpretation: runs differ in medium and duration; temporal RF stability is a diagnostic flag, not proof of a simulator fault.", fontsize=8, color="0.25")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.15, top=0.88, wspace=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
