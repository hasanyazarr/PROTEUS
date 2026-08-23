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
