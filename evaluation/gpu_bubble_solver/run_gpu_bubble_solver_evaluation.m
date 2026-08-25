function run_gpu_bubble_solver_evaluation(varargin)
%RUN_GPU_BUBBLE_SOLVER_EVALUATION Evaluate CPU/GPU pressure and solver agreement.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
parser = inputParser;
addParameter(parser, 'SettingsPath', fullfile(repoRoot, ...
    'simulation-settings', 'my_simulation_settings.mat'));
addParameter(parser, 'OutputDir', fullfile(repoRoot, ...
    'evaluation_results', 'gpu_bubble_solver'));
% Phase-step sweep. The RK4 substep count is chosen so that no substep
% advances more than this many radians of the fastest timescale, which is the
% only accuracy control the fixed-step solver has. Empty means "just the value
% in the settings", which is what a plain run does.
addParameter(parser, 'PhaseSteps', []);
parse(parser, varargin{:});

settingsPath = char(parser.Results.SettingsPath);
outputDir = char(parser.Results.OutputDir);
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

addpath(repoRoot)
PATHS = path_setup();
addpath(PATHS.AcousticModulePath)
addpath(PATHS.MicrobubblePath)
addpath(fullfile(PATHS.MicrobubblePath, 'functions'))
addpath(PATHS.StreamlineFunctions)
addpath(PATHS.GUIfunctions)
addpath(genpath(PATHS.kWavePath))
preflight(settingsPath, PATHS);
gpuDevice(1);
phaseSettings = normalize_settings_types(load(settingsPath, 'Microbubble'));
maxPhaseStep = resolve_gpu_rk4_max_phase_step(phaseSettings.Microbubble);
phaseSteps = resolve_phase_step_sweep(parser.Results.PhaseSteps, maxPhaseStep);
gpuPrecision = 'single';
if isfield(phaseSettings.Microbubble, 'GPUPrecision')
    gpuPrecision = char(phaseSettings.Microbubble.GPUPrecision);
end
gpuMaxStride = 6;
if isfield(phaseSettings.Microbubble, 'GPUMaxStride')
    gpuMaxStride = phaseSettings.Microbubble.GPUMaxStride;
end
gpuPressureInterp = 'pchip';
if isfield(phaseSettings.Microbubble, 'GPUPressureInterp')
    gpuPressureInterp = char(phaseSettings.Microbubble.GPUPressureInterp);
end

seed = 1729;
frequencies = [2.5e6, 6e6, 18e6];
pressures = [50e3, 200e3];
radii = [0.5e-6, 1e-6, 2.14e-6, 5e-6];
samplingRate = 250e6;
gpuRepeats = 3;

environment = collect_environment(repoRoot, settingsPath, seed);
environment.analytic_frequencies_hz = frequencies;
environment.analytic_pressures_pa = pressures;
environment.analytic_radii_m = radii;
environment.analytic_sampling_rate_hz = samplingRate;
environment.analytic_pulse_cycles = 2;
environment.analytic_truth_oversampling = 16;
environment.analytic_shell_model = 'Marmottant';
environment.gpu_repeats = gpuRepeats;
environment.gpu_batch_mode = 'auto';
environment.gpu_memory_fraction = 0.50;
environment.gpu_max_batch_size = 'inf';
environment.gpu_rk4_max_phase_step = maxPhaseStep;
environment.gpu_rk4_phase_step_sweep = phaseSteps;
environment.gpu_rk4_convergence_phase_step = phaseSteps(1) / 2;
environment.gpu_precision = gpuPrecision;
environment.gpu_max_stride = gpuMaxStride;
environment.gpu_pressure_interpolation = gpuPressureInterp;
environment.interpolation_strides = [1, 2, 4, 6];
environment.capture_solver = '3DG';
write_json(fullfile(outputDir, 'environment.json'), environment)
[interpolationMetrics, solverMetrics, analyticTimings, analyticDetails] = ...
    run_analytic_evaluation(frequencies, pressures, radii, ...
    samplingRate, gpuRepeats, phaseSteps, gpuPrecision, gpuMaxStride, ...
    gpuPressureInterp);
writetable(interpolationMetrics, fullfile(outputDir, ...
    'interpolation_metrics.csv'))
writetable(solverMetrics, fullfile(outputDir, 'solver_agreement.csv'))
writetable(analyticTimings, fullfile(outputDir, 'timings.csv'))
plot_interpolation_overlay(analyticDetails, fullfile(outputDir, ...
    'interpolation_overlay.png'))
plot_response_overlay(analyticDetails, fullfile(outputDir, ...
    'response_overlay.png'))

[capturePath, evaluationSettingsPath] = capture_real_pressure(...
    settingsPath, outputDir, PATHS, seed, maxPhaseStep, gpuPrecision, ...
    gpuMaxStride, gpuPressureInterp);
environment.evaluation_settings_path = evaluationSettingsPath;
environment.evaluation_settings_sha256 = command_output(sprintf( ...
    'sha256sum "%s" | cut -d " " -f 1', evaluationSettingsPath));
[captureEnvironment, capture] = capture_environment(capturePath);
captureEnvironmentFields = fieldnames(captureEnvironment);
for fieldIndex = 1:numel(captureEnvironmentFields)
    fieldName = captureEnvironmentFields{fieldIndex};
    environment.(fieldName) = captureEnvironment.(fieldName);
end
write_json(fullfile(outputDir, 'environment.json'), environment)
[realPressureMetrics, realTimings, realDetails] = ...
    run_real_pressure_evaluation(capture, gpuRepeats, phaseSteps);
timings = [analyticTimings; realTimings];

writetable(realPressureMetrics, fullfile(outputDir, ...
    'real_pressure_metrics.csv'))
writetable(timings, fullfile(outputDir, 'timings.csv'))
save(fullfile(outputDir, 'results.mat'), 'environment', ...
    'interpolationMetrics', 'solverMetrics', 'realPressureMetrics', ...
    'timings', 'analyticDetails', 'realDetails', 'capturePath', '-v7.3')
fprintf('\n=== Interpolation metrics ===\n'); disp(interpolationMetrics)
fprintf('\n=== CPU agreement reference metrics ===\n'); disp(solverMetrics)
fprintf('\n=== Real pressure metrics ===\n'); disp(realPressureMetrics)
fprintf('\n=== Timings ===\n'); disp(timings)
fprintf('\nEvaluation artifacts saved to %s\n', outputDir)
end

function preflight(settingsPath, PATHS)
if ~exist(settingsPath, 'file')
    error('gpuBubbleEvaluation:MissingSettings', ...
        'Settings file not found: %s', settingsPath)
end
if ~strcmp(version('-release'), '2025a')
    error('gpuBubbleEvaluation:UnsupportedMATLABRelease', ...
        'This evaluation requires MATLAB R2025a; found R%s.', ...
        version('-release'))
end
if ~license('test', 'Distrib_Computing_Toolbox')
    error('gpuBubbleEvaluation:MissingPCT', ...
        'Parallel Computing Toolbox is required.')
end
if gpuDeviceCount("available") == 0
    error('gpuBubbleEvaluation:MissingGPU', 'No available GPU was detected.')
end
cudaBinary = fullfile(PATHS.BinaryPath, 'kspaceFirstOrder-CUDA');
if ~exist(cudaBinary, 'file')
    error('gpuBubbleEvaluation:MissingKWaveBinary', ...
        'k-Wave CUDA binary not found: %s', cudaBinary)
end
% k-Wave ships kspaceFirstOrder3DC.m with "export LD_LIBRARY_PATH=;", which
% blanks the loader path before launching the binary. The CUDA build then
% cannot find libcuda.so.1 and aborts with "Insufficient CUDA driver version
% ... but 0.0 is installed" after the precomputation has already run.
solverWrapper = which('kspaceFirstOrder3DC');
if isempty(solverWrapper)
    error('gpuBubbleEvaluation:MissingKWaveWrapper', ...
        'kspaceFirstOrder3DC.m is not on the MATLAB path.')
end
if contains(fileread(solverWrapper), 'export LD_LIBRARY_PATH=;')
    error('gpuBubbleEvaluation:BlankCudaLibraryPath', ...
        ['%s blanks LD_LIBRARY_PATH before launching the CUDA binary, so ' ...
         'libcuda.so.1 cannot be found. Patch that line to export the CUDA ' ...
         'driver directories (for example /usr/local/cuda/lib64:' ...
         '/usr/lib64-nvidia:/usr/local/cuda/lib64/stubs) before running.'], ...
        solverWrapper)
end
% The CUDA solver rejects a power law exponent of exactly one, and this
% evaluation always runs on 3DG. Fail here rather than after the k-Wave
% precomputation has already been paid for.
attenuation = load(settingsPath, 'Medium');
if double(attenuation.Medium.AttenuationB) == 1
    error('gpuBubbleEvaluation:IllegalAttenuationPower', ...
        ['Medium.AttenuationB is exactly 1.0, which kspaceFirstOrder-CUDA ' ...
         'rejects ("Illegal value of alpha_power"). Set it slightly off 1 ' ...
         '(1.01 keeps the dispersion term disabled) before running.'])
end
settings = load(settingsPath, 'Geometry');
geometryFolder = fullfile(PATHS.GeometriesPath, settings.Geometry.Folder);
requiredGeometryFiles = {'vtu.mat', 'inlet.mat', 'GeometryProperties.mat'};
for i = 1:numel(requiredGeometryFiles)
    requiredPath = fullfile(geometryFolder, requiredGeometryFiles{i});
    if ~exist(requiredPath, 'file')
        error('gpuBubbleEvaluation:MissingGeometryData', ...
            'Required geometry file not found: %s', requiredPath)
    end
end
end

function phaseSteps = resolve_phase_step_sweep(requested, settingsPhaseStep)
%RESOLVE_PHASE_STEP_SWEEP Validate the sweep and order it finest first.

if isempty(requested)
    phaseSteps = settingsPhaseStep;
    return
end
if ~isnumeric(requested) || ~isreal(requested) || ...
        any(~isfinite(requested)) || any(requested <= 0)
    error('gpuBubbleEvaluation:InvalidPhaseSteps', ...
        'PhaseSteps must be finite positive phase limits in radians.')
end
% Ascending, so phaseSteps(1) is the finest and can act as the reference the
% coarser steps are measured against.
phaseSteps = unique(double(requested(:)'));
end

function environment = collect_environment(repoRoot, settingsPath, seed)
device = gpuDevice;
environment.created_at_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
environment.git_commit = command_output(sprintf( ...
    'git -C "%s" rev-parse HEAD', repoRoot));
environment.matlab_version = version;
environment.matlab_release = version('-release');
environment.gpu_name = device.Name;
environment.gpu_compute_capability = device.ComputeCapability;
environment.gpu_total_memory_bytes = double(device.TotalMemory);
environment.gpu_available_memory_bytes = double(device.AvailableMemory);
environment.cpu_dtype = 'double';
environment.random_seed = seed;
environment.settings_path = settingsPath;
environment.settings_sha256 = command_output(sprintf( ...
    'sha256sum "%s" | cut -d " " -f 1', settingsPath));
environment.cpu_reference_label = 'CPU agreement reference';
environment.gpu_solver = 'fixed-step RK4';
environment.cpu_solver = 'adaptive ode45';
environment.cpu_pressure_interpolation = 'pchip';
end

function output = command_output(command)
[status, output] = system(command);
if status ~= 0
    error('gpuBubbleEvaluation:CommandFailed', ...
        'Command failed: %s\n%s', command, output)
end
output = strtrim(output);
end

function [interpolationTable, solverTable, timingTable, details] = ...
        run_analytic_evaluation(frequencies, pressures, radii, ...
        samplingRate, gpuRepeats, phaseSteps, gpuPrecision, ...
        gpuMaxStride, gpuPressureInterp)
% PHASESTEPS is swept ascending, so the first entry is the finest and serves
% as the solver's own reference: the spread against it separates the RK4
% discretization error from everything else that differs from the CPU path
% (single precision, and the pressure interpolant on the coarse grid).
maxPhaseStep = phaseSteps(1);
interpolationStrides = [1, 2, 4, 6];
[liquid, gas] = getMaterialProperties();
liquid.ThermalModel = 'Prosperetti';
[bubble, shell] = make_bubbles_and_shells(radii, liquid);
interpolationRows = empty_interpolation_rows();
solverRows = empty_solver_rows();
timingRows = empty_timing_rows();
details = struct();

for frequency = frequencies
    for pressure = pressures
        pulse = make_analytic_pulse(frequency, pressure, samplingRate, ...
            numel(radii), maxPhaseStep);
        pulse.gpuPrecision = gpuPrecision;
        pulse.gpuMaxStride = gpuMaxStride;
        pulse.gpuPressureInterp = gpuPressureInterp;
        [truthTime, truthPressure] = analytic_truth_pressure(...
            frequency, pressure, pulse.t(end), samplingRate * 16);
        % The solver interpolates pressure across its strided output grid,
        % so the interpolation error is measured at the same spacings.
        for strideValue = interpolationStrides
            [linearPressure, pchipPressure] = interpolate_strided_pressure(...
                pulse, strideValue, truthTime);
            interpolationRows(end + 1) = interpolation_row(...
                frequency, pressure, 'linear', strideValue, ...
                truthPressure, linearPressure); %#ok<AGROW>
            interpolationRows(end + 1) = interpolation_row(...
                frequency, pressure, 'pchip', strideValue, ...
                truthPressure, pchipPressure); %#ok<AGROW>
        end

        % The CPU reference is independent of the phase step; run it once.
        cpuTimer = tic;
        cpuResponse = calcBubbleResponse(liquid, gas, shell, bubble, pulse);
        cpuSeconds = toc(cpuTimer);

        sweepResponses = cell(1, numel(phaseSteps));
        sweepInfo = cell(1, numel(phaseSteps));
        sweepSeconds = zeros(1, numel(phaseSteps));
        for phaseIndex = 1:numel(phaseSteps)
            sweptPulse = pulse;
            sweptPulse.rk4MaxPhaseStep = phaseSteps(phaseIndex);
            % One untimed call first: the first launch of a kernel pays for
            % its compilation, which would land on whichever step ran first.
            calcBubbleResponse_GPU(liquid, gas, shell, bubble, sweptPulse);
            gpuDurations = zeros(gpuRepeats, 1);
            for repeatIndex = 1:gpuRepeats
                gpuTimer = tic;
                [gpuResponse, ~, gpuSolverInfo] = calcBubbleResponse_GPU(...
                    liquid, gas, shell, bubble, sweptPulse);
                gpuDurations(repeatIndex) = toc(gpuTimer);
            end
            sweepResponses{phaseIndex} = gpuResponse;
            sweepInfo{phaseIndex} = gpuSolverInfo;
            sweepSeconds(phaseIndex) = median(gpuDurations);
        end
        finestResponse = sweepResponses{1};
        % The finest step stands in for "the GPU result" wherever a single one
        % is needed: the convergence check and the overlay plot.
        gpuResponse = sweepResponses{1};
        gpuSolverInfo = sweepInfo{1};

        convergence = repmat(empty_convergence_metrics(), 1, numel(radii));
        if frequency == 18e6 && pressure == 200e3
            convergenceBubbleIndex = find(radii == 0.5e-6, 1);
            refinedPulse = pulse;
            refinedPulse.rk4MaxPhaseStep = maxPhaseStep / 2;
            refinedResponse = calcBubbleResponse_GPU(...
                liquid, gas, shell, bubble, refinedPulse);
            convergence(convergenceBubbleIndex) = response_convergence(...
                radii(convergenceBubbleIndex), liquid.rho, ...
                gpuResponse(convergenceBubbleIndex), ...
                refinedResponse(convergenceBubbleIndex), ...
                maxPhaseStep, refinedPulse.rk4MaxPhaseStep);
        end

        for phaseIndex = 1:numel(phaseSteps)
            for bubbleIndex = 1:numel(radii)
                solverRows(end + 1) = solver_row(frequency, pressure, ...
                    radii(bubbleIndex), liquid.rho, ...
                    cpuResponse(bubbleIndex), ...
                    sweepResponses{phaseIndex}(bubbleIndex), ...
                    convergence(bubbleIndex), phaseSteps(phaseIndex), ...
                    finestResponse(bubbleIndex), phaseSteps(1)); %#ok<AGROW>
            end
            timingRows(end + 1) = timing_row('analytic', frequency, ...
                pressure, numel(radii), numel(pulse.t), cpuSeconds, ...
                sweepSeconds(phaseIndex), NaN, ...
                sweepInfo{phaseIndex}); %#ok<AGROW>
        end

        if frequency == 18e6 && pressure == 200e3
            solverStride = gpuSolverInfo.stride;
            [solverLinear, solverPchip] = interpolate_strided_pressure(...
                pulse, solverStride, truthTime);
            details.interpolation.t = truthTime;
            details.interpolation.truth = truthPressure;
            details.interpolation.linear = solverLinear;
            details.interpolation.pchip = solverPchip;
            details.interpolation.stride = solverStride;
            details.interpolation.frequency = frequency;
            details.interpolation.pressure = pressure;
            selectedBubble = 3;
            details.response.t = cpuResponse(selectedBubble).t;
            details.response.cpu = (cpuResponse(selectedBubble).R - ...
                radii(selectedBubble)) / radii(selectedBubble);
            details.response.gpu = (gpuResponse(selectedBubble).R - ...
                radii(selectedBubble)) / radii(selectedBubble);
            details.response.radius = radii(selectedBubble);
            details.response.phase_step = phaseSteps(1);
            details.response.frequency = frequency;
            details.response.pressure = pressure;
        end
    end
end
interpolationTable = struct2table(interpolationRows);
solverTable = struct2table(solverRows);
timingTable = struct2table(timingRows);
end

function pulse = make_analytic_pulse(...
    frequency, pressure, samplingRate, nBubbles, maxPhaseStep)
pulseDuration = 2 / frequency;
pulse.t = 0:(1 / samplingRate):(pulseDuration + 3e-6);
basePressure = analytic_hann_pressure(pulse.t, frequency, pressure);
pulse.p = repmat(basePressure, nBubbles, 1);
pulse.f = frequency;
pulse.w = 2 * pi * frequency;
pulse.fs = samplingRate;
pulse.tq = pulse.t;
pulse.dispProgress = false;
pulse.rk4MaxPhaseStep = maxPhaseStep;
end

function [linearPressure, pchipPressure] = interpolate_strided_pressure(...
    pulse, stride, truthTime)
% Reconstruct the driving pressure from the grid the solver integrates on.

coarseTime = pulse.t(1:stride:end);
coarsePressure = pulse.p(1, 1:stride:end);
if coarseTime(end) ~= pulse.t(end)
    coarseTime(end + 1) = pulse.t(end);
    coarsePressure(end + 1) = pulse.p(1, end);
end
linearPressure = interp1(coarseTime, coarsePressure, truthTime, 'linear', 0);
pchipPressure = interp1(coarseTime, coarsePressure, truthTime, 'pchip', 0);
end

function pressure = analytic_hann_pressure(time, frequency, amplitude)
pulseDuration = 2 / frequency;
pressure = zeros(size(time));
insidePulse = time >= 0 & time <= pulseDuration;
window = sin(pi * time(insidePulse) / pulseDuration).^2;
pressure(insidePulse) = amplitude * sin(...
    2 * pi * frequency * time(insidePulse)) .* window;
end

function [time, pressure] = analytic_truth_pressure(...
        frequency, amplitude, endTime, samplingRate)
time = 0:(1 / samplingRate):endTime;
pressure = analytic_hann_pressure(time, frequency, amplitude);
end

function [bubble, shell] = make_bubbles_and_shells(radii, liquid)
for i = numel(radii):-1:1
    bubble(i).R0 = radii(i);
    shellInput.model = 'Marmottant';
    shellInput.sig_0 = 10e-3;
    shell(i) = getShellProperties(bubble(i), shellInput, liquid);
end
end

function row = interpolation_row(...
    frequency, pressure, method, stride, truth, estimate)
peakTruth = max(abs(truth));
row.frequency_hz = frequency;
row.pressure_pa = pressure;
row.method = {method};
row.stride = stride;
row.nrmse = norm(estimate - truth) / max(norm(truth), eps);
row.max_error_over_peak = max(abs(estimate - truth)) / max(peakTruth, eps);
row.relative_peak_amplitude_error = ...
    abs(max(abs(estimate)) - peakTruth) / max(peakTruth, eps);
end

function row = solver_row(...
    frequency, pressure, radius, density, cpu, gpu, convergence, ...
    phaseStep, finest, finestPhaseStep)
cpuExcursion = (cpu.R - radius) / radius;
gpuExcursion = (gpu.R - radius) / radius;
cpuMassSource = 4 * pi * density * cpu.R.^2 .* cpu.Rdot;
gpuMassSource = 4 * pi * density * gpu.R.^2 .* gpu.Rdot;
row.reference = {'CPU agreement reference'};
row.rk4_max_phase_step = phaseStep;
row.frequency_hz = frequency;
row.pressure_pa = pressure;
row.radius_m = radius;
row.radius_excursion_relative_l2 = relative_l2(cpuExcursion, gpuExcursion);
row.radius_excursion_max_abs = max(abs(cpuExcursion - gpuExcursion));
row.rdot_relative_l2 = relative_l2(cpu.Rdot, gpu.Rdot);
row.rdot_max_abs_m_per_s = max(abs(cpu.Rdot - gpu.Rdot));
row.mass_source_relative_l2 = relative_l2(cpuMassSource, gpuMassSource);
row.mass_source_max_abs = max(abs(cpuMassSource - gpuMassSource));
row.peak_time_difference_s = peak_time_difference(...
    cpu.t, cpuExcursion, gpu.t, gpuExcursion);
row.gpu_all_finite = all(isfinite(gpu.R)) && all(isfinite(gpu.Rdot));
row.gpu_positive_radius = all(gpu.R > 0);
row.rk4_convergence_reference_phase_step = convergence.referencePhaseStep;
row.rk4_convergence_refined_phase_step = convergence.refinedPhaseStep;
row.rk4_convergence_radius_relative_l2 = convergence.radiusRelativeL2;
row.rk4_convergence_rdot_relative_l2 = convergence.rdotRelativeL2;
row.rk4_convergence_mass_source_relative_l2 = ...
    convergence.massSourceRelativeL2;
% Against the finest phase step in the sweep: the solver's own discretization
% error, with the precision and interpolant differences divided out.
finestExcursion = (finest.R - radius) / radius;
finestMassSource = 4 * pi * density * finest.R.^2 .* finest.Rdot;
row.sweep_finest_phase_step = finestPhaseStep;
row.sweep_radius_excursion_relative_l2 = relative_l2(...
    finestExcursion, gpuExcursion);
row.sweep_mass_source_relative_l2 = relative_l2(...
    finestMassSource, gpuMassSource);
end

function metrics = response_convergence(...
    radius, density, reference, refined, referencePhaseStep, refinedPhaseStep)
referenceExcursion = (reference.R - radius) / radius;
refinedExcursion = (refined.R - radius) / radius;
referenceMassSource = 4 * pi * density * reference.R.^2 .* reference.Rdot;
refinedMassSource = 4 * pi * density * refined.R.^2 .* refined.Rdot;
metrics.referencePhaseStep = referencePhaseStep;
metrics.refinedPhaseStep = refinedPhaseStep;
metrics.radiusRelativeL2 = relative_l2(refinedExcursion, referenceExcursion);
metrics.rdotRelativeL2 = relative_l2(refined.Rdot, reference.Rdot);
metrics.massSourceRelativeL2 = relative_l2(...
    refinedMassSource, referenceMassSource);
end

function metrics = empty_convergence_metrics()
metrics.referencePhaseStep = NaN;
metrics.refinedPhaseStep = NaN;
metrics.radiusRelativeL2 = NaN;
metrics.rdotRelativeL2 = NaN;
metrics.massSourceRelativeL2 = NaN;
end

function value = relative_l2(reference, estimate)
value = norm(estimate(:) - reference(:)) / max(norm(reference(:)), eps);
end

function difference = peak_time_difference(tReference, reference, tEstimate, estimate)
[~, referenceIndex] = max(abs(reference));
[~, estimateIndex] = max(abs(estimate));
difference = abs(tReference(referenceIndex) - tEstimate(estimateIndex));
end

function [capturePath, temporarySettingsPath] = ...
        capture_real_pressure(settingsPath, outputDir, PATHS, seed, ...
        maxPhaseStep, gpuPrecision, gpuMaxStride, gpuPressureInterp)
% The GUI saves numeric fields as integer or single classes, which the
% streamline and acoustic modules cannot mix with double time arrays.
settings = normalize_settings_types(load(settingsPath));
requiredVariables = {'Acquisition', 'Geometry', 'Medium', 'Microbubble', ...
    'SimulationParameters', 'Transducer', 'Transmit'};
for i = 1:numel(requiredVariables)
    if ~isfield(settings, requiredVariables{i})
        error('gpuBubbleEvaluation:InvalidSettings', ...
            'Settings file is missing variable %s.', requiredVariables{i})
    end
end
settings.Acquisition.NumberOfFrames = 1;
settings.Acquisition.StartFrame = 1;
settings.Acquisition.EndFrame = 1;
settings.Acquisition.Continue = false;
settings.Acquisition.ParallelTracking = false;
seededBubbleCount = 100;
requestedBubbleCount = 25;
settings.Microbubble.Number = seededBubbleCount;
settings.Microbubble.GPURK4MaxPhaseStep = maxPhaseStep;
settings.Microbubble.GPUPrecision = gpuPrecision;
settings.Microbubble.GPUMaxStride = gpuMaxStride;
settings.Microbubble.GPUPressureInterp = gpuPressureInterp;
settings.Medium.Save = true;
settings.SimulationParameters.Solver = '3DG';
settings.SimulationParameters.DeviceNumber = 0;
runId = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyyMMdd_HHmmss'));
groundTruthName = ['gpu_bubble_eval_' runId];
resultName = ['gpu_bubble_eval_' runId];
capturePath = fullfile(outputDir, 'real_pressure_capture.mat');
settings.SimulationParameters.EvaluationCapture.Enabled = true;
settings.SimulationParameters.EvaluationCapture.OutputPath = capturePath;
settings.SimulationParameters.EvaluationCapture.StopAfterCapture = true;
settings.SimulationParameters.EvaluationCapture.SeededBubbleCount = ...
    seededBubbleCount;
settings.SimulationParameters.EvaluationCapture.RequestedBubbleCount = ...
    requestedBubbleCount;
temporarySettingsPath = fullfile(outputDir, 'evaluation_settings.mat');
save(temporarySettingsPath, '-struct', 'settings', '-v7.3')
rng(seed, 'twister')
generate_streamlines(settings.Geometry, settings.Microbubble, ...
    settings.Acquisition, PATHS, groundTruthName, false)
main_RF(temporarySettingsPath, groundTruthName, resultName, false, 1, 1, 0)
if ~exist(capturePath, 'file')
    error('gpuBubbleEvaluation:CaptureFailed', ...
        'PROTEUS did not produce the expected pressure capture.')
end
end

function [environment, capture] = capture_environment(capturePath)
loaded = load(capturePath, 'capture');
capture = loaded.capture;
environment.capture_effective_k_max = capture.kgrid.k_max;
environment.capture_hybrid_simulation = capture.hybrid_simulation;
environment.capture_seeded_bubble_count = capture.seeded_bubble_count;
environment.capture_valid_bubble_count = capture.valid_bubble_count;
environment.capture_selected_bubble_count = capture.selected_bubble_count;
environment.capture_solver = capture.solver;
environment.capture_device_number = capture.device_number;
end

function [metricTable, timingTable, details] = ...
        run_real_pressure_evaluation(capture, gpuRepeats, phaseSteps)
% The arm that answers the production question: real radii, the real driving
% pressure, and the real sample count, swept over the phase step. PHASESTEPS
% is ascending, so its first entry is the finest and doubles as the solver's
% own reference.
[~, sortedIndices] = sort(capture.radii);
nBubbles = numel(sortedIndices);
expectedBubbleCount = capture.selected_bubble_count;
if nBubbles ~= expectedBubbleCount
    error('gpuBubbleEvaluation:UnexpectedCapturedBubbleCount', ...
        ['Real-pressure evaluation requires %d captured bubbles according ' ...
         'to capture metadata; found %d.'], expectedBubbleCount, nBubbles)
end
positions = unique(round(linspace(1, nBubbles, 4)));
selectedIndices = sortedIndices(positions);
subsetPressure = capture.sensed_p(selectedIndices, :);
subsetRadii = capture.radii(selectedIndices);

cpuConfig = capture.Microbubble;
cpuConfig.UseGPU = false;
cpuConfig.UseParfor = 'off';
cpuConfig.BatchSize = numel(selectedIndices);
cpuTimer = tic;
[cpuMassSource, cpuInfo] = compute_bubble_mass_source(...
    subsetPressure, subsetRadii, capture.kgrid, capture.Medium, ...
    cpuConfig, capture.Transmit);
cpuSeconds = toc(cpuTimer);

gpuConfig = capture.Microbubble;
gpuConfig.UseGPU = true;
gpuConfig.UseParfor = 'off';
gpuConfig.GPUBatchSize = 'auto';
gpuConfig.GPUMemoryFraction = 0.50;
gpuConfig.GPUMaxBatchSize = inf;
sweepMassSource = cell(1, numel(phaseSteps));
sweepInfo = cell(1, numel(phaseSteps));
sweepSeconds = zeros(1, numel(phaseSteps));
for phaseIndex = 1:numel(phaseSteps)
    sweptConfig = gpuConfig;
    sweptConfig.GPURK4MaxPhaseStep = phaseSteps(phaseIndex);
    % Untimed warm-up so kernel compilation does not land on one step.
    compute_bubble_mass_source(subsetPressure, subsetRadii, ...
        capture.kgrid, capture.Medium, sweptConfig, capture.Transmit);
    gpuDurations = zeros(gpuRepeats, 1);
    for repeatIndex = 1:gpuRepeats
        gpuTimer = tic;
        [gpuMassSource, subsetGpuInfo] = compute_bubble_mass_source(...
            subsetPressure, subsetRadii, capture.kgrid, capture.Medium, ...
            sweptConfig, capture.Transmit);
        gpuDurations(repeatIndex) = toc(gpuTimer);
    end
    sweepMassSource{phaseIndex} = gpuMassSource;
    sweepInfo{phaseIndex} = subsetGpuInfo;
    sweepSeconds(phaseIndex) = median(gpuDurations);
end
subsetGpuInfo = sweepInfo{1};
subsetGpuMedian = sweepSeconds(1);
finestMassSource = sweepMassSource{1};

metricRows = empty_real_pressure_rows();
for phaseIndex = 1:numel(phaseSteps)
    gpuMassSource = sweepMassSource{phaseIndex};
    for i = 1:numel(selectedIndices)
        row.reference = {'CPU agreement reference'};
        row.rk4_max_phase_step = phaseSteps(phaseIndex);
        row.capture_index = selectedIndices(i);
        row.radius_m = subsetRadii(i);
        row.mass_source_relative_l2 = relative_l2(...
            cpuMassSource(i, :), gpuMassSource(i, :));
        row.mass_source_max_abs = max(abs(...
            cpuMassSource(i, :) - gpuMassSource(i, :)));
        row.peak_time_difference_s = peak_time_difference(...
            capture.t_kwave, cpuMassSource(i, :), ...
            capture.t_kwave, gpuMassSource(i, :));
        row.gpu_all_finite = all(isfinite(gpuMassSource(i, :)));
        row.sweep_finest_phase_step = phaseSteps(1);
        row.sweep_mass_source_relative_l2 = relative_l2(...
            finestMassSource(i, :), gpuMassSource(i, :));
        metricRows(end + 1) = row; %#ok<AGROW>
    end
end

% The full captured population at every phase step: the throughput half of
% the trade, at the bubble count and sample count a production frame uses.
allPressure = capture.sensed_p;
allRadii = capture.radii;
allSweepSeconds = zeros(1, numel(phaseSteps));
allSweepInfo = cell(1, numel(phaseSteps));
for phaseIndex = 1:numel(phaseSteps)
    sweptConfig = gpuConfig;
    sweptConfig.GPURK4MaxPhaseStep = phaseSteps(phaseIndex);
    compute_bubble_mass_source(allPressure, allRadii, capture.kgrid, ...
        capture.Medium, sweptConfig, capture.Transmit);
    allGpuDurations = zeros(gpuRepeats, 1);
    for repeatIndex = 1:gpuRepeats
        gpuTimer = tic;
        [allGpuMassSource, allGpuInfo] = compute_bubble_mass_source(...
            allPressure, allRadii, capture.kgrid, capture.Medium, ...
            sweptConfig, capture.Transmit);
        allGpuDurations(repeatIndex) = toc(gpuTimer);
    end
    allSweepSeconds(phaseIndex) = median(allGpuDurations);
    allSweepInfo{phaseIndex} = allGpuInfo;
end
allGpuInfo = allSweepInfo{1};

timingRows = empty_timing_rows();
for phaseIndex = 1:numel(phaseSteps)
    timingRows(end + 1) = timing_row('real_pressure_subset', NaN, NaN, ...
        numel(selectedIndices), size(subsetPressure, 2), cpuSeconds, ...
        sweepSeconds(phaseIndex), sweepInfo{phaseIndex}.batchSize, ...
        sweepInfo{phaseIndex}); %#ok<AGROW>
    timingRows(end + 1) = timing_row('real_pressure_all', NaN, NaN, ...
        nBubbles, size(allPressure, 2), NaN, allSweepSeconds(phaseIndex), ...
        allSweepInfo{phaseIndex}.batchSize, ...
        allSweepInfo{phaseIndex}); %#ok<AGROW>
end

details.selected_indices = selectedIndices;
details.cpu_info = cpuInfo;
details.subset_gpu_info = subsetGpuInfo;
details.all_gpu_info = allGpuInfo;
details.all_gpu_mass_source_finite = all(isfinite(allGpuMassSource), 'all');
metricTable = struct2table(metricRows);
timingTable = struct2table(timingRows);
end

function row = timing_row(source, frequency, pressure, nBubbles, nSamples, ...
        cpuSeconds, gpuSeconds, batchSize, solverInfo)
row.source = {source};
row.frequency_hz = frequency;
row.pressure_pa = pressure;
row.number_of_bubbles = nBubbles;
row.number_of_samples = nSamples;
row.cpu_seconds = cpuSeconds;
row.gpu_median_seconds = gpuSeconds;
if isnan(cpuSeconds); row.speedup = NaN; else; row.speedup = cpuSeconds / gpuSeconds; end
row.batch_size = batchSize;
[substeps, maxAngularFrequency, maxPhaseStep, actualPhaseStep, stride] = ...
    timing_diagnostics(solverInfo);
row.rk4_substeps = max(substeps, [], 'all', 'omitnan');
row.rk4_substeps_min = min(substeps, [], 'all', 'omitnan');
row.rk4_max_angular_frequency_rad_s = maxAngularFrequency;
row.rk4_max_phase_step = maxPhaseStep;
row.rk4_actual_phase_step_max = actualPhaseStep;
row.stride_max = stride;
end

function [substeps, maxAngularFrequency, maxPhaseStep, actualPhaseStep, ...
        stride] = timing_diagnostics(solverInfo)
if isempty(solverInfo) || isempty(fieldnames(solverInfo))
    substeps = NaN;
    maxAngularFrequency = NaN;
    maxPhaseStep = NaN;
    actualPhaseStep = NaN;
    stride = NaN;
elseif isfield(solverInfo, 'substeps')
    substeps = solverInfo.substeps;
    maxAngularFrequency = solverInfo.maxAngularFrequency;
    maxPhaseStep = solverInfo.maxPhaseStep;
    actualPhaseStep = solverInfo.actualPhaseStep;
    stride = solverInfo.stride;
else
    substeps = solverInfo.rk4SubstepsPerBatch;
    maxAngularFrequency = max(...
        solverInfo.rk4MaxAngularFrequencyPerBatch, [], 'all');
    maxPhaseStep = solverInfo.rk4MaxPhaseStep;
    actualPhaseStep = max(...
        solverInfo.rk4ActualPhaseStepPerBatch, [], 'all');
    stride = max(solverInfo.stridePerBatch, [], 'all');
end
end

function rows = empty_interpolation_rows()
template = struct('frequency_hz', 0, 'pressure_pa', 0, 'method', {{}}, ...
    'stride', 0, 'nrmse', 0, 'max_error_over_peak', 0, ...
    'relative_peak_amplitude_error', 0);
rows = repmat(template, 0, 1);
end

function rows = empty_solver_rows()
template = struct('reference', {{}}, 'rk4_max_phase_step', 0, ...
    'frequency_hz', 0, 'pressure_pa', 0, 'radius_m', 0, ...
    'radius_excursion_relative_l2', 0, 'radius_excursion_max_abs', 0, ...
    'rdot_relative_l2', 0, 'rdot_max_abs_m_per_s', 0, ...
    'mass_source_relative_l2', 0, 'mass_source_max_abs', 0, ...
    'peak_time_difference_s', 0, 'gpu_all_finite', false, ...
    'gpu_positive_radius', false, ...
    'rk4_convergence_reference_phase_step', 0, ...
    'rk4_convergence_refined_phase_step', 0, ...
    'rk4_convergence_radius_relative_l2', 0, ...
    'rk4_convergence_rdot_relative_l2', 0, ...
    'rk4_convergence_mass_source_relative_l2', 0, ...
    'sweep_finest_phase_step', 0, ...
    'sweep_radius_excursion_relative_l2', 0, ...
    'sweep_mass_source_relative_l2', 0);
rows = repmat(template, 0, 1);
end

function rows = empty_real_pressure_rows()
template = struct('reference', {{}}, 'rk4_max_phase_step', 0, ...
    'capture_index', 0, 'radius_m', 0, ...
    'mass_source_relative_l2', 0, 'mass_source_max_abs', 0, ...
    'peak_time_difference_s', 0, 'gpu_all_finite', false, ...
    'sweep_finest_phase_step', 0, 'sweep_mass_source_relative_l2', 0);
rows = repmat(template, 0, 1);
end

function rows = empty_timing_rows()
template = struct('source', {{}}, 'frequency_hz', 0, 'pressure_pa', 0, ...
    'number_of_bubbles', 0, 'number_of_samples', 0, ...
    'cpu_seconds', 0, 'gpu_median_seconds', 0, 'speedup', 0, ...
    'batch_size', 0, 'rk4_substeps', 0, 'rk4_substeps_min', 0, ...
    'rk4_max_angular_frequency_rad_s', 0, 'rk4_max_phase_step', 0, ...
    'rk4_actual_phase_step_max', 0, 'stride_max', 0);
rows = repmat(template, 0, 1);
end

function write_json(path, value)
fileId = fopen(path, 'w');
if fileId < 0
    error('gpuBubbleEvaluation:OutputOpenFailed', ...
        'Could not open output file: %s', path)
end
cleanup = onCleanup(@() fclose(fileId));
fwrite(fileId, jsonencode(value, 'PrettyPrint', true), 'char');
clear cleanup
end

function plot_interpolation_overlay(details, outputPath)
figureHandle = figure('Visible', 'off');
plot(details.interpolation.t * 1e6, details.interpolation.truth / 1e3, ...
    'k-', 'LineWidth', 1.5); hold on
plot(details.interpolation.t * 1e6, details.interpolation.linear / 1e3, 'r--')
plot(details.interpolation.t * 1e6, details.interpolation.pchip / 1e3, 'b:')
xlabel('Time (\mus)'); ylabel('Pressure (kPa)')
legend('Analytic truth', 'Linear', 'PCHIP', 'Location', 'best'); grid on
title(sprintf('18 MHz, 200 kPa pressure interpolation (stride %d)', ...
    details.interpolation.stride))
exportgraphics(figureHandle, outputPath, 'Resolution', 160); close(figureHandle)
end

function plot_response_overlay(details, outputPath)
figureHandle = figure('Visible', 'off');
plot(details.response.t * 1e6, details.response.cpu, ...
    'k-', 'LineWidth', 1.5); hold on
plot(details.response.t * 1e6, details.response.gpu, 'r--')
xlabel('Time (\mus)'); ylabel('(R - R_0) / R_0')
legend('CPU ode45', sprintf('GPU RK4 (%.3g rad/substep)', ...
    details.response.phase_step), 'Location', 'best'); grid on
title('2.14 \mum bubble response at 18 MHz, 200 kPa')
exportgraphics(figureHandle, outputPath, 'Resolution', 160); close(figureHandle)
end
