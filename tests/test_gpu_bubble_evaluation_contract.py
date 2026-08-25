from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_main_rf_capture_is_opt_in_and_can_stop_after_first_pressure():
    source = (ROOT / "acoustic-module" / "main_RF.m").read_text()

    assert "SimulationParameters.EvaluationCapture" in source
    assert "capture_sensed_pressure_if_requested" in source
    assert "StopAfterCapture" in source
    assert "if stopAfterCapture" in source
    assert "return" in source
    assert "bubble_counts" in source
    assert source.index(
        "validate_evaluation_capture_yield_if_requested"
    ) < source.index("run_simulation(run_param, kgrid")


def test_capture_contains_replay_inputs_for_mass_source_evaluation():
    source = (ROOT / "acoustic-module" / "main_RF.m").read_text()

    for field in (
        "capture.sensed_p",
        "capture.radii",
        "capture.t_kwave",
        "capture.kgrid",
        "capture.Medium",
        "capture.Microbubble",
        "capture.Transmit",
        "capture.hybrid_simulation",
        "capture.solver",
        "capture.seeded_bubble_count",
        "capture.valid_bubble_count",
        "capture.selected_bubble_count",
    ):
        assert field in source


def test_evaluator_exposes_one_entrypoint_and_standard_grid():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "function run_gpu_bubble_solver_evaluation(varargin)" in source
    assert "frequencies = [2.5e6, 6e6, 18e6];" in source
    assert "pressures = [50e3, 200e3];" in source
    assert "radii = [0.5e-6, 1e-6, 2.14e-6, 5e-6];" in source
    assert "samplingRate = 250e6;" in source
    assert "gpuRepeats = 3;" in source


def test_evaluator_writes_all_planned_artifacts():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    for artifact in (
        "environment.json",
        "interpolation_metrics.csv",
        "solver_agreement.csv",
        "real_pressure_metrics.csv",
        "timings.csv",
        "results.mat",
        "interpolation_overlay.png",
        "response_overlay.png",
    ):
        assert artifact in source


def test_evaluator_runs_analytic_and_captured_pressure_paths():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "run_analytic_evaluation" in source
    assert "capture_real_pressure" in source
    assert "run_real_pressure_evaluation" in source
    assert "CPU agreement reference" in source
    assert "calculate_substeps" not in source
    assert "rk4_convergence_mass_source_relative_l2" in source
    assert "capture.selected_bubble_count" in source
    assert "settings.Microbubble.GPURK4MaxPhaseStep = maxPhaseStep;" in source
    assert "settings.Microbubble.GPURK4MaxPhaseStep = 0.25;" not in source


def test_capture_uses_gpu_solver_and_exact_deterministic_bubble_subset():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "seededBubbleCount = 100;" in source
    assert "requestedBubbleCount = 25;" in source
    assert "settings.SimulationParameters.Solver = '3DG';" in source
    assert "settings.SimulationParameters.DeviceNumber = 0;" in source
    assert "gpuDevice(1);" in source
    assert "resultName, false, 1, 1, 0)" in source


def test_mass_source_exposes_actual_batch_diagnostics():
    source = (
        ROOT / "acoustic-module" / "compute_bubble_mass_source.m"
    ).read_text()

    assert "function [mass_source, runInfo] = compute_bubble_mass_source" in source
    assert "runInfo.useGPU = useGPU;" in source
    assert "runInfo.batchSize = batchSize;" in source
    assert "runInfo.numberOfBatches = Nbatch;" in source
    assert "runInfo.numberOfOutputSamples = numberOfOutputSamples;" in source
    assert "runInfo.rk4SubstepsPerBatch" in source
    assert "runInfo.rk4MaxAngularFrequencyPerBatch" in source
    assert "runInfo.rk4ActualPhaseStepPerBatch" in source
    assert "runInfo.stridePerBatch" in source
    assert "runInfo.gpuPrecision" in source


def test_colab_launcher_has_a_runnable_help_command():
    launcher = (
        ROOT / "evaluation" / "gpu_bubble_solver" / "run_colab_evaluation.py"
    )

    completed = subprocess.run(
        [sys.executable, str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--settings" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_preflight_detects_the_blank_cuda_library_path():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "gpuBubbleEvaluation:BlankCudaLibraryPath" in source
    assert "export LD_LIBRARY_PATH=;" in source
    assert "which('kspaceFirstOrder3DC')" in source


def test_preflight_rejects_the_attenuation_power_the_cuda_solver_refuses():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "gpuBubbleEvaluation:IllegalAttenuationPower" in source
    assert "double(attenuation.Medium.AttenuationB) == 1" in source
    # The check has to precede the run, not trail the k-Wave precomputation.
    assert source.index("IllegalAttenuationPower") < source.index(
        "function [interpolationTable, solverTable, timingTable, details] ="
    )


def test_evaluation_normalizes_gui_settings_types_before_running():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "normalize_settings_types(load(settingsPath))" in source
    assert "normalize_settings_types(load(settingsPath, 'Microbubble'))" in source
    assert "settings = load(settingsPath);" not in source


def test_interpolation_error_is_measured_at_the_strides_the_solver_uses():
    source = (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()

    assert "interpolationStrides = [1, 2, 4, 6];" in source
    assert "for strideValue = interpolationStrides" in source
    assert "interpolate_strided_pressure(" in source
    assert "coarseTime = pulse.t(1:stride:end);" in source
    assert "row.stride = stride;" in source
    assert "solverStride = gpuSolverInfo.stride;" in source


def test_colab_launcher_runs_matlab_behavior_tests_before_evaluation():
    source = (
        ROOT / "evaluation" / "gpu_bubble_solver" / "run_colab_evaluation.py"
    ).read_text()

    assert "test_gpu_bubble_solver_helpers.m" in source


def test_generated_evaluation_results_are_ignored():
    ignore = (ROOT / ".gitignore").read_text().splitlines()

    assert "/evaluation_results/" in ignore


def test_readme_documents_the_single_colab_cell_and_evidence_boundary():
    readme = (
        ROOT / "evaluation" / "gpu_bubble_solver" / "README.md"
    ).read_text()

    assert "run_colab_evaluation.py" in readme
    assert "run_colab_evaluation.py" in readme
    assert "A100" in readme
    assert "does not define scientific pass/fail thresholds" in readme


def evaluator_source():
    return (
        ROOT
        / "evaluation"
        / "gpu_bubble_solver"
        / "run_gpu_bubble_solver_evaluation.m"
    ).read_text()


def test_phase_step_is_a_swept_axis_not_a_single_settings_value():
    """The RK4 phase budget is the fixed-step solver's only accuracy control,
    and the Aug 23 tightening from ~0.75 to 0.25 rad cost 18x in run time. The
    evaluator has to be able to price that trade, which means running the same
    case at several budgets rather than at whatever the settings happen to say.
    """
    source = evaluator_source()

    assert "addParameter(parser, 'PhaseSteps', []);" in source
    assert "function phaseSteps = resolve_phase_step_sweep(" in source
    # Absent the parameter, a plain run behaves exactly as before.
    assert "phaseSteps = settingsPhaseStep;" in source
    assert "gpuBubbleEvaluation:InvalidPhaseSteps" in source


def test_sweep_is_ordered_so_the_finest_step_is_the_reference():
    source = evaluator_source()

    assert "phaseSteps = unique(double(requested(:)'));" in source
    assert "finestResponse = sweepResponses{1};" in source
    assert "finestMassSource = sweepMassSource{1};" in source


def test_both_arms_sweep_the_phase_step():
    """The analytic arm prices accuracy against the CPU reference; the real
    pressure arm prices it at the bubble count and sample count a production
    frame actually uses. A sweep of only the first would not answer the
    production question."""
    source = evaluator_source()

    assert "run_analytic_evaluation(frequencies, pressures, radii, ...\n" \
           "    samplingRate, gpuRepeats, phaseSteps," in source
    assert "run_real_pressure_evaluation(capture, gpuRepeats, phaseSteps);" in source
    assert "sweptPulse.rk4MaxPhaseStep = phaseSteps(phaseIndex);" in source
    assert "sweptConfig.GPURK4MaxPhaseStep = phaseSteps(phaseIndex);" in source


def test_swept_rows_record_which_phase_step_produced_them():
    """A sweep whose rows cannot be told apart is not a sweep."""
    source = evaluator_source()

    assert "row.rk4_max_phase_step = phaseStep;" in source
    assert "row.rk4_max_phase_step = phaseSteps(phaseIndex);" in source
    assert "'rk4_max_phase_step', 0, ...\n    'frequency_hz'" in source
    # Error against the finest step isolates RK4 discretization from the
    # precision and interpolant differences that the CPU comparison mixes in.
    assert "row.sweep_mass_source_relative_l2" in source
    assert "row.sweep_radius_excursion_relative_l2" in source
    assert "'sweep_finest_phase_step', 0" in source


def test_each_swept_point_is_timed_after_a_warm_up():
    """Kernel compilation on the first launch would otherwise be charged to
    whichever phase step happened to run first."""
    source = evaluator_source()

    assert source.count("gpuDurations(repeatIndex) = toc(gpuTimer);") == 2
    assert "allSweepSeconds(phaseIndex) = median(allGpuDurations);" in source
    assert "sweepSeconds(phaseIndex) = median(gpuDurations);" in source
