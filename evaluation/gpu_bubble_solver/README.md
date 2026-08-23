# GPU bubble solver evaluation

This Colab-oriented pipeline compares the original CPU `ode45`/PCHIP
microbubble response with the fixed-step GPU RK4/linear implementation. It
also measures linear and PCHIP interpolation against a known analytic pulse
and replays pressure captured from a one-frame PROTEUS incident-field run.

## Prerequisites

- Google Colab with an NVIDIA A100 GPU.
- MATLAB R2025a with online licensing and Parallel Computing Toolbox.
- The canonical PROTEUS Colab setup completed, including the k-Wave CUDA
  binary, geometry data, and `simulation-settings/my_simulation_settings.mat`.

## Single Colab cell

```python
%cd /content/PROTEUS
!git fetch origin main
!git checkout main
!git pull --ff-only origin main
!python3 evaluation/gpu_bubble_solver/run_colab_evaluation.py \
    --settings /content/PROTEUS/simulation-settings/my_simulation_settings.mat \
    --output-dir /content/PROTEUS/evaluation_results/gpu_bubble_solver
```

The launcher prints the CSV tables and displays the interpolation and response
plots inline. Raw outputs include environment provenance, all metric tables,
the real-pressure capture, and a MATLAB results file.
Before starting the A100 evaluation, the launcher runs
`tests/matlab/test_gpu_bubble_solver_helpers.m` and stops if a behavior test
fails.

The evaluator always uses the k-Wave `3DG` CUDA solver on device 0. It seeds
100 deterministic bubbles, captures the first 25 valid in-grid bubbles, and
runs the all-bubble timing on exactly those 25 pressure traces. The capture
records the effective hybrid/full mass-source filter cutoff rather than the
unmodified k-Wave grid cutoff.

GPU RK4 integration uses the fastest bubble natural frequency, damping rate,
and transmit frequency when selecting substeps. The maximum phase step is read
from `Microbubble.GPURK4MaxPhaseStep` with a legacy default of `0.25` radians
and is recorded with the actual per-batch substep counts and output stride. The
18 MHz, 200 kPa, 0.5 micrometer case is repeated at half that phase step to
report a numerical convergence indicator. A complex, non-finite, or
non-positive-radius state stops the run with frame/pulse/batch/bubble context;
invalid states are never clamped, including the ones the fused kernels would
otherwise clamp away mid-substep.

The solver keeps the throughput settings the production runs were tuned for and
reports both of them alongside every timing row:

| Setting | Default | Effect |
|---|---|---|
| `Microbubble.GPUPrecision` | `'single'` | Floating-point class of the GPU integration. `'double'` trades throughput for accuracy. |
| `Microbubble.GPUMaxStride` | `6` | Upper bound on the coarse output grid. `1` integrates every microbubble sample and disables spline interpolation. |
| `Microbubble.GPURK4MaxPhaseStep` | `0.25` | Maximum phase advanced per RK4 substep [rad]. Substeps scale with the stride. |

`Medium.AttenuationB` must not be exactly `1.0`: `kspaceFirstOrder-CUDA`
refuses that power law exponent outright, and the evaluation always runs on
`3DG`. Preflight rejects it before the k-Wave precomputation. `1.01` is the
smallest change that clears the check and still keeps the dispersion term
disabled, which is what `define_medium` does for exponents near 1 anyway.

For local contract checks with MATLAB R2025a:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch \
  "run('tests/matlab/test_gpu_bubble_solver_helpers.m')"
```

`interpolation_metrics.csv` measures linear and PCHIP reconstruction of the
driving pressure against the closed-form pulse at strides 1, 2, 4, and 6, so
the error is reported at the spacing the solver actually integrates on rather
than at the raw microbubble sampling rate. The overlay plot shows the stride
the solver selected for the 18 MHz, 200 kPa case.

This first evaluation does not define scientific pass/fail thresholds. The
CPU result is labeled as an agreement reference rather than ground truth. Use
the recorded A100 evidence to decide whether the GPU pressure interpolation
must be changed from linear to PCHIP.
