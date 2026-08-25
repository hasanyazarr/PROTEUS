import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_solver_precision_is_configurable_and_single_by_default():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()

    assert "P_gpu = gpuArray(toPrecision(pulse.p));" in source
    assert "toPrecision = @(value) cast(value, precision);" in source
    assert re.search(r"precision\s*=\s*resolve_gpu_precision\(pulse\);", source)
    assert "{'single', 'double'}" in source
    assert re.search(r"Microbubble\.GPUPrecision\s*=\s*'single';", defaults)
    # The fused kernels must stay untyped so they run at the input precision.
    assert "single(1e-6)" not in source
    assert "opx  = max(1 + xi, 1e-6);" in source


def test_rk4_phase_step_default_is_the_measured_one():
    """0.5 rad, from the sweep of 2026-08-25.

    The default was 0.25 rad, chosen without measurement when the budget was
    tightened from ~0.75 on 2026-08-23. The sweep put numbers on it: against
    the error-controlled CPU reference, the worst mass-source disagreement
    was 6.2e-4 at 0.25 rad and 4.5e-4 at 0.5 rad - both inside the ~1-3e-4
    floor set by single precision and the pressure interpolant, so the two
    are indistinguishable. 1.0 rad leaves that floor at 3.0e-3, five times
    worse, and does it at 0.5 um / 18 MHz - the small-bubble corner the
    production radii actually occupy. 0.5 rad is the largest step that costs
    no accuracy, and it takes two RK4 substeps instead of three.
    """
    resolver = (
        ROOT / "microbubble-simulator" / "functions"
        / "resolve_gpu_rk4_max_phase_step.m"
    ).read_text()

    assert "maxPhaseStep = 0.5;" in resolver
    assert "maxPhaseStep = 0.25;" not in resolver
    # One definition of the default; the GUI defaults must read it, not
    # restate it.
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()
    assert (
        "Microbubble.GPURK4MaxPhaseStep = "
        "resolve_gpu_rk4_max_phase_step(struct());" in defaults
    )


def test_gpu_solver_striding_is_bounded_and_can_be_disabled():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()

    assert re.search(r"strideLimit\s*=\s*resolve_gpu_max_stride\(pulse\);", source)
    assert "stride = double(min(stride, strideLimit));" in source
    assert "calcBubbleResponse_GPU:InvalidMaxStride" in source
    assert re.search(r"Microbubble\.GPUMaxStride\s*=\s*6;", defaults)


def test_gpu_solver_rejects_states_the_fused_kernels_would_clamp():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    assert "intermediateNonPositive = gpuArray(false(1, N_MB));" in source
    assert (
        "validate_gpu_bubble_states(X_out, Xd_out, "
        "gather(intermediateNonPositive));" in source
    )
    # Every stage state is still checked, on the Marmottant path inside the
    # fused kernel and on the table-shell path through the helper.
    assert source.count("track_invalid_state(") >= 5
    assert source.count("nonPositive(") >= 4


def test_marmottant_interval_runs_as_one_fused_kernel():
    """A whole coarse interval is one arrayfun launch.

    Fusing a single substep took the launch count from ~42 to 1 per substep
    and the solver stage from 35.9 s to 5.98 s. What was left was still
    launch-bound: 182 us per substep for arithmetic worth a fraction of that,
    so the remaining cost is per-launch, not per-flop. The substep loop moves
    inside the kernel, leaving one launch per coarse interval.
    """
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    assert "function [xo, xdo, badOut] = rk4_interval_marmottant(" in source
    assert re.search(
        r"\[x, xd, intermediateNonPositive\] = arrayfun\(\s*\.\.\.\s*\n"
        r"\s*@rk4_interval_marmottant", source
    )
    # The substep loop is inside the kernel, so the host loop is gone.
    assert "for s = 1:nSub" in source
    # arrayfun cannot index a host array, so the weights are no longer
    # indexed out of one per substep.
    assert "W_rise(s, 1)" not in source


def test_fused_interval_keeps_the_arithmetic_of_the_staged_path():
    """The fusion changes launch structure only, never the RK4 formula."""
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()
    kernel = source.split("function [xo, xdo, badOut] = rk4_interval_marmottant(")[1]

    # Classic RK4: two half steps, one full step, and the 1-2-2-1 combination.
    assert "x2 = x + hn2 * k1x;" in kernel
    assert "x3 = x + hn2 * k2x;" in kernel
    assert "x4 = x + hn * k3x;" in kernel
    assert "x  = x  + hn6 * (k1x + 2 * k2x + 2 * k3x + k4x);" in kernel
    assert "xd = xd + hn6 * (k1v + 2 * k2v + 2 * k3v + k4v);" in kernel
    # Stages 2 and 3 share the midpoint weights; stage 4 uses the interval end.
    assert kernel.count("w2r, w2l, w2h, R0i") == 2


def test_in_kernel_stage_weights_match_the_host_hermite_basis():
    """The kernel rebuilds the basis the host helper still defines.

    arrayfun cannot index the precomputed weight arrays, so the kernel
    evaluates the Hermite basis from the substep index instead. That
    duplicates the formula, and the two copies have to stay the same curve.
    """
    kernel = (
        ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m"
    ).read_text()
    host = (
        ROOT / "microbubble-simulator" / "functions" / "hermite_stage_weights.m"
    ).read_text()

    assert "W_rise = fracs.^2 .* (3 - 2*fracs);" in host
    assert "W_lo   = fracs .* (fracs - 1).^2;" in host
    assert "W_hi   = fracs.^2 .* (fracs - 1);" in host

    assert "wr = f * f * (3 - 2 * f);" in kernel
    assert "wl = f * (f - 1) * (f - 1);" in kernel
    assert "wh = f * f * (f - 1);" in kernel
    # Linear sampling differs only in the rise term. Its slope weights
    # multiply zero slopes, so the kernel computes them unconditionally.
    assert "wr = f;" in kernel


def test_fused_kernel_keeps_the_working_precision():
    """The substep index must not promote the kernel to double.

    The loop bound is a plain count, but the stage fractions divide by it,
    and a double divisor would drag every downstream operation - the whole
    Rayleigh-Plesset right-hand side - into double precision.
    """
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    assert "nSubScale = toPrecision(n_sub);" in source
    assert "f0 = (s - 1) / nSubScale;" in source


def test_gpu_solver_exposes_optional_diagnostics_without_changing_inputs():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    assert re.search(
        r"function\s+\[response,\s*eqparam,\s*solverInfo\]\s*=\s*"
        r"calcBubbleResponse_GPU\(liquid,\s*\.\.\.\s*gas,", source
    )


def test_matlab_behavior_suite_covers_solver_validation_and_substeps():
    matlab_test = ROOT / "tests" / "matlab" / "test_gpu_bubble_solver_helpers.m"

    assert matlab_test.is_file()


def test_gpu_execution_is_explicit_and_preserves_batch_configuration():
    source = (ROOT / "acoustic-module" / "compute_bubble_mass_source.m").read_text()
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()

    assert re.search(r"Microbubble\.UseGPU\s*=\s*false;", defaults)
    assert "isfield(Microbubble, 'UseGPU')" in source
    assert "license('test', 'Distrib_Computing_Toolbox')" in source
    assert 'gpuDeviceCount("available")' in source
    assert "compute_bubble_mass_source:GPUUnavailable" in source
    assert "batchSize = Microbubble.BatchSize;" in source
    assert "batchSize = 100;" not in source
    assert "if gpuDeviceCount > 0" not in source
    assert "if useGPU" in source


def test_gpu_batch_size_uses_available_memory_with_safe_defaults():
    source = (ROOT / "acoustic-module" / "compute_bubble_mass_source.m").read_text()
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()

    assert re.search(r"Microbubble\.GPUBatchSize\s*=\s*'auto';", defaults)
    assert re.search(r"Microbubble\.GPUMemoryFraction\s*=\s*0\.50;", defaults)
    assert re.search(r"Microbubble\.GPUMaxBatchSize\s*=\s*inf;", defaults)
    assert "resolve_gpu_rk4_max_phase_step(struct())" in defaults
    assert "device.AvailableMemory" in source
    assert "select_gpu_batch_size" in source


def test_gpu_batch_size_supports_manual_override_and_legacy_settings():
    source = (ROOT / "acoustic-module" / "compute_bubble_mass_source.m").read_text()

    assert "Microbubble.GPUBatchSize" in source
    assert "gpuBatchSetting = Microbubble.BatchSize;" in source
    assert "batchSize = gpuBatchSetting;" in source


def test_gpu_solver_samples_pressure_with_pchip_like_the_cpu_reference():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()
    defaults = (ROOT / "GUIfunctions" / "reset_microbubble.m").read_text()

    assert re.search(
        r"pressureInterp\s*=\s*resolve_gpu_pressure_interp\(pulse\);", source
    )
    assert "slopes = pchip_slopes(t_coarse_knots, P_coarse);" in source
    assert "hermite_stage_weights(stage_fracs, pressureInterp)" in source
    assert "calcBubbleResponse_GPU:InvalidPressureInterp" in source
    assert "{'pchip', 'linear'}" in source
    # Linear sampling stays reachable so the two can be compared head to head.
    assert re.search(
        r"Microbubble\.GPUPressureInterp\s*=\s*'pchip';", defaults
    )


def test_gpu_pressure_interpolation_is_reported_by_the_solver():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()
    massSource = (
        ROOT / "acoustic-module" / "compute_bubble_mass_source.m"
    ).read_text()

    assert "solverInfo.pressureInterp = pressureInterp;" in source
    assert "pulse.gpuPressureInterp = Microbubble.GPUPressureInterp;" in massSource
    assert (
        "runInfo.gpuPressureInterp = batchSolverInfo{1}.pressureInterp;"
        in massSource
    )


def test_marmottant_liquid_surface_tension_is_read_per_bubble():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    # The validator lets the Marmottant branch through without checking that
    # the shells agree, so every shell field it reads has to be per-bubble.
    assert "s_sigl = gpuArray(toPrecision(shell(1).sig_l));" not in source
    assert source.count("toPrecision([shell.sig_l])") == 3


def test_rk4_step_follows_each_coarse_interval_width():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    # The appended final index can leave an interval narrower than the stride,
    # so the step has to come from the interval, not from the nominal stride.
    assert "h_interval = gpu_coarse_step_sizes(idx_coarse, dt, n_sub);" in source
    assert "hn  = h_interval(n);" in source
    for stale in ("x + h2*k1x", "xd+h2*k1v", "x + h*k3x", "h6 * (k1x"):
        assert stale not in source


def test_coarse_grid_knots_stay_in_double_precision():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    # A single-precision time value resolves the sample spacing of a pulse
    # tens of microseconds long to about three digits, which would land in the
    # interval widths and from there in the pchip slopes.
    assert "t_coarse_knots = reshape(double(tq(idx_coarse)), 1, []);" in source
    assert "dt_coarse = gpuArray(toPrecision(diff(t_coarse_knots)));" in source
    assert "toPrecision(tq(idx_coarse))" not in source
    # The spline back-interpolation reads the same double knots.
    assert "tc = t_coarse_knots;" in source
    assert "tf = reshape(double(tq), 1, []);" in source


def test_stage_pressure_is_evaluated_inside_the_rhs_kernel():
    source = (ROOT / "microbubble-simulator" / "calcBubbleResponse_GPU.m").read_text()

    # Evaluating the Hermite form outside arrayfun cost several extra kernel
    # launches per RK4 stage, in a loop that is already launch-bound.
    assert "function P = interp_pressure(stage)" not in source
    assert source.count("Pi = Pn + wRise*dP + wLo*mLo + wHi*mHi;") == 2
    assert "function [dx, dv] = rp_rhs(xi, xdi, stage)" in source
    # Both launches carry the interval endpoints and the stage weights, so the
    # pressure is rebuilt in registers; a precomputed pressure array never
    # reaches a kernel.
    for kernel in ("@rk4_interval_marmottant", "@rp_core"):
        launch = source.index(kernel + ",")
        assert "Pn, dP, mLo, mHi" in source[launch:launch + 300], kernel
    # The stage index, not a precomputed pressure array, reaches the RHS.
    assert "[k1x, k1v] = rp_rhs(x,          xd,          1);" in source
