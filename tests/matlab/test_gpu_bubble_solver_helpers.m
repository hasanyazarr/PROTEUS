repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'microbubble-simulator'));
addpath(fullfile(repoRoot, 'microbubble-simulator', 'functions'));
addpath(fullfile(repoRoot, 'acoustic-module'));

%% RK4 substeps use both the bubble and transmit angular frequencies.
pulse.t = [0, 4e-9];
pulse.tq = pulse.t;
pulse.p = zeros(1, 2);
pulse.w = 2*pi*18e6;
eqparam.omega_0 = 50e6;
[nSub, info] = calculate_gpu_rk4_substeps(eqparam, pulse, 0.25);
assert(nSub == 2);
assert(info.maxAngularFrequency == pulse.w);
assert(info.maxPhaseStep == 0.25);
assert(info.dimensionalStep == 4e-9);

eqparam.omega_0 = 300e6;
[eqparam.delta] = deal(0.5);
[nSub, info] = calculate_gpu_rk4_substeps(eqparam, pulse, 0.25);
assert(nSub == 5);
assert(info.maxAngularFrequency == 300e6);

eqparam.omega_0 = 50e6;
eqparam.delta = 8;
[nSub, info] = calculate_gpu_rk4_substeps(eqparam, pulse, 0.25);
assert(nSub == 7);
assert(info.maxAngularFrequency == 400e6);

%% A strided output grid lengthens the step the substeps have to cover.
eqparam = struct('omega_0', 50e6);
[nSubUnstrided, infoUnstrided] = calculate_gpu_rk4_substeps(eqparam, pulse, 0.25);
[nSubStrided, infoStrided] = calculate_gpu_rk4_substeps(eqparam, pulse, 0.25, 4);
assert(infoStrided.dimensionalStep == 4 * infoUnstrided.dimensionalStep);
assert(nSubStrided == 4 * nSubUnstrided);
assert(infoStrided.actualPhaseStep <= 0.25);

try
    calculate_gpu_rk4_substeps(eqparam, pulse, 0.25, 0);
    error('test:NoError', 'A non-positive stride must be rejected.');
catch exception
    assert(strcmp(exception.identifier, 'calcBubbleResponse_GPU:InvalidStride'));
end

%% The phase-step default has one resolver and respects explicit settings.
assert(resolve_gpu_rk4_max_phase_step(struct()) == 0.25);
phaseSettings.GPURK4MaxPhaseStep = 0.125;
assert(resolve_gpu_rk4_max_phase_step(phaseSettings) == 0.125);

%% Solver input validation rejects mismatched pressure and time grids.
shell = make_marmottant_shell();
validate_gpu_bubble_solver_inputs(pulse, shell);

badPulse = pulse;
badPulse.tq = [0, 8e-9];
assert_error(@() validate_gpu_bubble_solver_inputs(badPulse, shell), ...
    'calcBubbleResponse_GPU:MismatchedTimeGrids');

badPulse = pulse;
badPulse.p = zeros(1, 3);
assert_error(@() validate_gpu_bubble_solver_inputs(badPulse, shell), ...
    'calcBubbleResponse_GPU:PressureLengthMismatch');

badPulse = pulse;
badPulse.t = [0, 4e-9, 9e-9];
badPulse.tq = badPulse.t;
badPulse.p = zeros(1, 3);
assert_error(@() validate_gpu_bubble_solver_inputs(badPulse, shell), ...
    'calcBubbleResponse_GPU:NonUniformTimeGrid');

%% Shell validation names unsupported and inconsistent models explicitly.
badShell = shell;
badShell.model = 'Unknown';
assert_error(@() validate_gpu_bubble_solver_inputs(pulse, badShell), ...
    'calcBubbleResponse_GPU:UnsupportedShellModel');

mixedShell = [shell, shell];
mixedShell(2).model = 'Segers';
twoBubblePulse = pulse;
twoBubblePulse.p = zeros(2, 2);
assert_error(@() validate_gpu_bubble_solver_inputs(twoBubblePulse, mixedShell), ...
    'calcBubbleResponse_GPU:MixedShellModels');

tableShell = make_table_shell();
differentTableShell = [tableShell, tableShell];
differentTableShell(2).sig.Values(2) = 0.5;
assert_error(@() validate_gpu_bubble_solver_inputs(...
    twoBubblePulse, differentTableShell), ...
    'calcBubbleResponse_GPU:InconsistentShellTables');

%% Gathered and intermediate invalid states are rejected without clamping.
validate_gpu_bubble_states([0; 0.1], [0; 0], false(1, 1));
assert_error(@() validate_gpu_bubble_states([0; 0.1 + 1i], [0; 0], ...
    false(1, 1)), 'calcBubbleResponse_GPU:ComplexState');
assert_error(@() validate_gpu_bubble_states([0; -1], [0; 0], ...
    true(1, 1)), 'calcBubbleResponse_GPU:NonPositiveRadius');

%% An empty GPU frame is a valid no-op even without a local GPU.
emptyMicrobubble.UseGPU = true;
emptyMicrobubble.SamplingRate = 250e6;
emptyGrid.dt = 4e-9;
[emptyMassSource, emptyInfo] = compute_bubble_mass_source(...
    zeros(0, 8, 'single'), zeros(0, 1), emptyGrid, struct(), ...
    emptyMicrobubble, struct());
assert(isequal(size(emptyMassSource), [0, 8]));
assert(strcmp(class(emptyMassSource), 'single'));
assert(emptyInfo.batchSize == 1);
assert(emptyInfo.numberOfBatches == 0);

%% Capture and production share the same effective filter cutoff.
kgrid.k_max = 12;
kgrid.dt = 4e-9;
medium.SpeedOfSoundMinimum = 1480;
assert(get_bubble_filter_kmax(kgrid, medium, false) == 12);
expectedHybrid = pi / kgrid.dt / medium.SpeedOfSoundMinimum;
assert(get_bubble_filter_kmax(kgrid, medium, true) == expectedHybrid);

%% Adding the optional solver diagnostics output preserves old call syntax.
assert(nargout('calcBubbleResponse_GPU') == 3);

disp('GPU bubble solver helper tests passed.')

%% Coarse step sizes cover the grid exactly, short last interval included.
dt = 1e-3;
nSub = 3;
% 10 samples at stride 4 leaves a final interval one sample wide.
idxCoarse = int32([1, 5, 9, 10]);
hInterval = gpu_coarse_step_sizes(idxCoarse, dt, nSub);
assert(numel(hInterval) == numel(idxCoarse) - 1);
assert(abs(hInterval(1) - 4*dt/nSub) < eps);
assert(abs(hInterval(end) - dt/nSub) < eps, ...
    'The short final interval must not step the full stride.');
% Integrating every substep must land exactly on the last sample.
totalAdvance = sum(hInterval) * nSub;
assert(abs(totalAdvance - (double(idxCoarse(end)) - 1)*dt) < 1e-12);

% An evenly divided grid keeps a single uniform step.
uniformH = gpu_coarse_step_sizes(int32([1, 5, 9]), dt, nSub);
assert(all(abs(uniformH - 4*dt/nSub) < eps));

% The step class follows dt, so the single-precision path stays single.
assert(isa(gpu_coarse_step_sizes(int32([1, 3]), single(dt), nSub), 'single'));

assert_error(@() gpu_coarse_step_sizes(int32(1), dt, nSub), ...
    'calcBubbleResponse_GPU:InvalidCoarseGrid');
assert_error(@() gpu_coarse_step_sizes(int32([1, 5, 5]), dt, nSub), ...
    'calcBubbleResponse_GPU:InvalidCoarseGrid');
assert_error(@() gpu_coarse_step_sizes(int32([1, 5]), dt, 0), ...
    'calcBubbleResponse_GPU:InvalidSubstepCount');

%% Pchip slopes reproduce MATLAB's own pchip interpolant.
x = [0, 0.7, 1.1, 2.6, 3.0, 4.4];
Y = [sin(3*x); exp(-x); [0, 0, 1, 1, 1, 2]];
m = pchip_slopes(x, Y);
assert(isequal(size(m), size(Y)));
for row = 1:size(Y, 1)
    for k = 1:(numel(x) - 1)
        H = x(k+1) - x(k);
        sFrac = (0.05:0.05:0.95)';
        [wRise, wLo, wHi] = hermite_stage_weights(sFrac, 'pchip');
        ours = Y(row, k) + wRise*(Y(row, k+1) - Y(row, k)) ...
            + wLo*(H*m(row, k)) + wHi*(H*m(row, k+1));
        expected = pchip(x, Y(row, :), x(k) + sFrac*H);
        assert(max(abs(ours - expected)) < 1e-12, ...
            'Hermite reconstruction must match pchip (row %d, span %d).', ...
            row, k);
    end
end

%% Pchip slopes flatten at extrema and stay exact on two knots.
mono = pchip_slopes([0, 1, 2], [0, 1, 0]);
assert(mono(2) == 0);
pair = pchip_slopes([0, 2], [1, 5]);
assert(isequal(pair, [2, 2]));

%% Hermite weights collapse to linear interpolation on request.
fracs = [0; 0.25; 0.5; 1];
[wRise, wLo, wHi] = hermite_stage_weights(fracs, 'linear');
assert(isequal(wRise, fracs));
assert(all(wLo == 0) && all(wHi == 0));

% At the knots the pchip weights must reproduce the endpoint values exactly.
[wRise, wLo, wHi] = hermite_stage_weights([0; 1], 'pchip');
assert(isequal(wRise, [0; 1]));
assert(all(wLo == 0) && all(wHi == 0));

function shell = make_marmottant_shell()
shell.model = 'Marmottant';
shell.chi = 0.55;
shell.Rb = 1e-6;
shell.sig_l = 0.072;
shell.Ks = 1e-8;
shell.sig_0 = 0.01;
end

function shell = make_table_shell()
shell.model = 'SegersTable';
shell.sig = griddedInterpolant([0, 1, 2], [0, 0.02, 0.072]);
shell.A_N = 1;
shell.A_m1 = 0;
shell.A_m2 = 2;
shell.sig_l = 0.072;
shell.Ks = 1e-8;
shell.sig_0 = 0.01;
end

function assert_error(callback, expectedIdentifier)
try
    callback();
catch exception
    assert(strcmp(exception.identifier, expectedIdentifier), ...
        'Expected %s, received %s.', expectedIdentifier, exception.identifier);
    return
end
error('test_gpu_bubble_solver_helpers:ExpectedError', ...
    'Expected error %s was not raised.', expectedIdentifier);
end
