#!/usr/bin/env python3
"""Run the MATLAB GPU bubble solver evaluation and display its artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def matlab_quote(value: Path) -> str:
    return str(value).replace("'", "''")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the PROTEUS GPU bubble solver evaluation in Colab."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=repo_root / "simulation-settings" / "my_simulation_settings.mat",
        help="PROTEUS MATLAB settings file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "evaluation_results" / "gpu_bubble_solver",
        help="Directory for CSV, JSON, MAT, and PNG artifacts.",
    )
    parser.add_argument(
        "--matlab",
        type=Path,
        default=Path("/usr/local/MATLAB/R2025a/bin/matlab"),
        help="MATLAB executable.",
    )
    return parser.parse_args()


def display_artifacts(output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print(f"Artifacts saved to {output_dir}")
        return

    for name in (
        "interpolation_metrics.csv",
        "solver_agreement.csv",
        "real_pressure_metrics.csv",
        "timings.csv",
    ):
        path = output_dir / name
        print(f"\n=== {name} ===")
        print(pd.read_csv(path).to_string(index=False))

    try:
        from IPython.display import Image, display
    except ImportError:
        return

    for name in ("interpolation_overlay.png", "response_overlay.png"):
        display(Image(filename=str(output_dir / name)))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    settings = args.settings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    matlab = args.matlab.expanduser()

    if not settings.is_file():
        raise SystemExit(f"Settings file not found: {settings}")
    if not matlab.is_file():
        raise SystemExit(f"MATLAB executable not found: {matlab}")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = repo_root / "evaluation" / "gpu_bubble_solver"
    matlab_test = repo_root / "tests" / "matlab" / "test_gpu_bubble_solver_helpers.m"
    matlab_command = (
        f"cd('{matlab_quote(repo_root)}'); "
        f"run('{matlab_quote(matlab_test)}'); "
        f"addpath('{matlab_quote(evaluation_dir)}'); "
        "run_gpu_bubble_solver_evaluation("
        f"'SettingsPath','{matlab_quote(settings)}',"
        f"'OutputDir','{matlab_quote(output_dir)}');"
    )
    subprocess.run(
        [str(matlab), "-batch", matlab_command, "-licmode", "onlinelicensing"],
        check=True,
    )
    display_artifacts(output_dir)


if __name__ == "__main__":
    main()
