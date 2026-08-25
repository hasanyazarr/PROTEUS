function results = run_solver_throughput_benchmark(varargin)
%RUN_SOLVER_THROUGHPUT_BENCHMARK Time both bubble solvers against bubble count.
%
%   The phase-step sweep timed the GPU solver at whatever bubble count its
%   capture happened to hold. Production asks a different question: at
%   Microbubble.Number, which solver is faster? That is what decides whether
%   Microbubble.UseGPU should be on.
%
%   The pressure capture is replayed rather than recomputed. A k-Wave capture
%   costs roughly twenty minutes, and these timings do not depend on what the
%   pressure contains - only on how many rows it has. Bubbles are drawn
%   cyclically from the captured population so the radius distribution stays
%   representative: radius sets the RK4 substep count on the GPU and the step
%   adaptation of ode45 on the CPU.
%
%   Reading a CPU timing needs the batch count. compute_bubble_mass_source
%   only reaches parfor when UseParfor is 'auto' and there are more than
%   eight batches, so with the production BatchSize of 100 the CPU path runs
%   single-threaded up to 800 bubbles and only then goes parallel.
%
%   Budget roughly fifteen minutes: the CPU arm dominates, and its worst
%   point is the largest count that still misses the parfor threshold.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
parser = inputParser;
addParameter(parser, 'CapturePath', fullfile(repoRoot, ...
    'evaluation_results', 'gpu_bubble_solver', 'real_pressure_capture.mat'));
addParameter(parser, 'OutputDir', fullfile(repoRoot, ...
    'evaluation_results', 'solver_throughput'));
% 200 is the production count in the v4 notebook; the rest bracket it and
% span the batch count where the CPU path turns parfor on.
addParameter(parser, 'BubbleCounts', [25, 50, 100, 200, 400, 800]);
% The GPU is fast enough to repeat; the CPU arm is the runtime of this whole
% benchmark, and ode45 on fixed input is deterministic enough not to need it.
addParameter(parser, 'GpuRepeats', 3);
addParameter(parser, 'CpuRepeats', 1);
% Empty means the repository default, which is what production will resolve.
addParameter(parser, 'PhaseStep', []);
parse(parser, varargin{:});

capturePath = char(parser.Results.CapturePath);
outputDir = char(parser.Results.OutputDir);
bubbleCounts = sort(double(parser.Results.BubbleCounts));
gpuRepeats = double(parser.Results.GpuRepeats);
cpuRepeats = double(parser.Results.CpuRepeats);

if ~exist(capturePath, 'file')
    error('gpuBubbleThroughput:MissingCapture', ...
        ['Pressure capture not found: %s\n' ...
         'Run run_gpu_bubble_solver_evaluation once to produce one, or ' ...
         'point CapturePath at a saved copy.'], capturePath)
end
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

addpath(repoRoot)
PATHS = path_setup();
addpath(PATHS.AcousticModulePath)
addpath(PATHS.MicrobubblePath)
addpath(fullfile(PATHS.MicrobubblePath, 'functions'))

if ~license('test', 'Distrib_Computing_Toolbox')
    error('gpuBubbleThroughput:MissingPCT', ...
        'Parallel Computing Toolbox is required.')
end
if gpuDeviceCount("available") == 0
    error('gpuBubbleThroughput:MissingGPU', 'No available GPU was detected.')
end
gpuDevice(1);

loaded = load(capturePath, 'capture');
capture = loaded.capture;
capturedCount = numel(capture.radii);

phaseStep = parser.Results.PhaseStep;
if isempty(phaseStep)
    phaseStep = resolve_gpu_rk4_max_phase_step(struct());
end

% Production settings, with only the solver choice varied between the arms.
cpuConfig = capture.Microbubble;
cpuConfig.UseGPU = false;
gpuConfig = capture.Microbubble;
gpuConfig.UseGPU = true;
gpuConfig.GPURK4MaxPhaseStep = phaseStep;

% Starting the pool costs tens of seconds and would otherwise land on
% whichever bubble count happened to reach parfor first.
pool = gcp('nocreate');
if isempty(pool); pool = parpool; end

% One untimed run per arm: the first GPU launch pays for kernel compilation
% and the first CPU call pays for JIT. Neither is a per-count cost, so both
% are paid once here rather than inside the sweep.
[warmPressure, warmRadii] = draw_population(capture, min(bubbleCounts));
warm_up_solver(warmPressure, warmRadii, capture, cpuConfig);
warm_up_solver(warmPressure, warmRadii, capture, gpuConfig);

rows = empty_rows();
for count = bubbleCounts
    [pressure, radii] = draw_population(capture, count);

    [cpuSeconds, cpuInfo] = time_solver(pressure, radii, capture, ...
        cpuConfig, cpuRepeats);
    rows(end + 1) = benchmark_row(count, 'cpu_ode45', cpuSeconds, ...
        cpuInfo, capture, phaseStep, pool.NumWorkers); %#ok<AGROW>

    [gpuSeconds, gpuInfo] = time_solver(pressure, radii, capture, ...
        gpuConfig, gpuRepeats);
    rows(end + 1) = benchmark_row(count, 'gpu_rk4', gpuSeconds, ...
        gpuInfo, capture, phaseStep, pool.NumWorkers); %#ok<AGROW>

    fprintf(['%5d bubbles: CPU %8.2f s (%d batches) | GPU %8.2f s | ' ...
        'GPU faster by %5.2fx\n'], count, cpuSeconds, ...
        cpuInfo.numberOfBatches, gpuSeconds, cpuSeconds / gpuSeconds);
end

results = struct2table(rows);
writetable(results, fullfile(outputDir, 'solver_throughput.csv'))

environment = struct();
environment.created_at_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
environment.capture_path = capturePath;
environment.captured_bubble_count = capturedCount;
environment.bubble_counts = bubbleCounts;
environment.gpu_rk4_max_phase_step = phaseStep;
environment.gpu_repeats = gpuRepeats;
environment.cpu_repeats = cpuRepeats;
environment.pool_workers = pool.NumWorkers;
environment.use_parfor_setting = char(string(cpuConfig.UseParfor));
environment.batch_size_setting = cpuConfig.BatchSize;
write_benchmark_json(fullfile(outputDir, 'environment.json'), environment)

fprintf('\nThroughput results saved to %s\n', outputDir)
end


function [pressure, radii] = draw_population(capture, n)
% Draw N bubbles cyclically from the captured population, so the radius
% distribution - and with it the substep count and ode45's step adaptation -
% stays the one the capture measured.

idx = mod(0:(n - 1), numel(capture.radii)) + 1;
pressure = capture.sensed_p(idx, :);
radii = capture.radii(idx);
end


function warm_up_solver(pressure, radii, capture, config)
% An untimed call, so kernel compilation and JIT do not land on a timed run.

compute_bubble_mass_source(pressure, radii, capture.kgrid, ...
    capture.Medium, config, capture.Transmit);
end


function [seconds, runInfo] = time_solver(pressure, radii, capture, ...
        config, repeats)
% Median of REPEATS timed calls.

durations = zeros(repeats, 1);
for i = 1:repeats
    timer = tic;
    [~, runInfo] = compute_bubble_mass_source(pressure, radii, ...
        capture.kgrid, capture.Medium, config, capture.Transmit);
    durations(i) = toc(timer);
end
seconds = median(durations);
end


function row = benchmark_row(count, solver, seconds, runInfo, capture, ...
        phaseStep, workers)
row.bubble_count = count;
row.solver = {solver};
row.seconds_median = seconds;
row.seconds_per_bubble = seconds / count;
row.batch_size = runInfo.batchSize;
row.number_of_batches = runInfo.numberOfBatches;
% 'auto' only reaches parfor above eight batches, and the GPU path never
% does - workers sharing one device add contention rather than throughput.
row.parfor_eligible = ~runInfo.useGPU && ...
    strcmp(char(string(capture.Microbubble.UseParfor)), 'auto') && ...
    runInfo.numberOfBatches > 8;
row.pool_workers = workers;
row.number_of_output_samples = runInfo.numberOfOutputSamples;
row.rk4_max_phase_step = phaseStep;
if isempty(runInfo.rk4SubstepsPerBatch)
    row.rk4_substeps = NaN;
    row.stride = NaN;
else
    row.rk4_substeps = max(runInfo.rk4SubstepsPerBatch, [], 'all');
    row.stride = max(runInfo.stridePerBatch, [], 'all');
end
end


function rows = empty_rows()
rows = benchmark_row(0, '', 0, empty_run_info_stub(), ...
    struct('Microbubble', struct('UseParfor', 'off')), 0, 0);
rows(1) = [];
end


function runInfo = empty_run_info_stub()
runInfo = struct('batchSize', 0, 'numberOfBatches', 0, 'useGPU', false, ...
    'numberOfOutputSamples', 0, 'rk4SubstepsPerBatch', [], ...
    'stridePerBatch', []);
end


function write_benchmark_json(path, data)
fid = fopen(path, 'w');
if fid == -1
    error('gpuBubbleThroughput:CannotWriteJson', ...
        'Could not open %s for writing.', path)
end
closeFile = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(data, 'PrettyPrint', true));
end
