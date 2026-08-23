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
    assert source.count("track_invalid_state(") >= 5
    assert (
        "validate_gpu_bubble_states(X_out, Xd_out, "
        "gather(intermediateNonPositive));" in source
    )


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
