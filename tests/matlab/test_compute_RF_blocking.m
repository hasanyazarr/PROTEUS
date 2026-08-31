% Blocking the receive path must not change the RF data.
%
% compute_RF is blocked along the element axis and chunked along the sensor
% axis, because v10's lambda/8 grid takes four of its arrays past the
% intmax('int32') element cap on a gpuArray. Neither split can run on a GPU
% here, so run_param.RFBlockElements and run_param.RFSensorChunkElements
% force the same partitions on the CPU.
%
% The element split is compared bit for bit: an element's RF line is the
% quadrature sum over its own integration points, so blocking re-blocks the
% arithmetic without reordering any sum. The sensor split cannot claim that
% -- it turns one row's product into partial sums added chunk by chunk, and
% reassociating a sum in double before the cast to single is allowed to move
% the last bit -- so it is compared to a tolerance, and the tolerance is
% tight enough that a real error could not hide under it.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));
run_log('reset');

rng(11);

N_el     = 7;      % deliberately not a multiple of any block size below
N_int    = 3;
N_sensor = 29;
M        = 40;

% Built from scratch, not assigned into: scripts run through run() share the
% base workspace, and a run_param left behind by another test would silently
% override the partition this one is measuring.
Grid = struct('dt', 1/50e6);
Transducer = struct();

Transducer.integration_points = randn(N_el, N_int, 3);
Transducer.integration_receive_apodization = rand(N_el, N_int);
Transducer.integration_receive_delays = 1e-8 * rand(N_el, N_int);
Transducer.SamplingRate = 250e6;
Transducer.ReceiveImpulseResponse = randn(1, 9);

% Sensor weights with a contiguous support per integration point, as
% update_sensor produces: one point spreads over neighbouring grid points.
N_points = N_el*N_int;
support  = 4;
i_idx = repmat((1:N_points)', 1, support);
j_idx = mod((0:support-1) + (0:N_points-1)', N_sensor) + 1;
v_idx = rand(N_points, support);

% One element that no sensor point carries, so the empty-block path runs.
i_keep = i_idx(:); j_keep = j_idx(:); v_keep = v_idx(:);
dead_rows = N_el:N_el:N_points;          % element N_el, every integration pt
alive = ~ismember(i_keep, dead_rows);
sensor_weights = sparse(i_keep(alive), j_keep(alive), v_keep(alive), ...
    N_points, N_sensor);

sensor_data.p = single(randn(N_sensor, M));

run_param = struct('DATA_CAST_RF', 'single', ...
    'RFBlockElements', [], 'RFSensorChunkElements', []);

N = rf_signal_length(Transducer, Grid, M, 'single');
assert(N > M, 'the delays did not extend the record; the test is weaker');

for weightCase = 1:2
    if weightCase == 1
        Transducer.integration_weights = 1.7;              % scalar
    else
        Transducer.integration_weights = rand(N_el, N_int); % per point
    end

    run_param.RFBlockElements = [];
    run_param.RFSensorChunkElements = [];
    whole = compute_RF(Transducer, sensor_data, sensor_weights, ...
        Grid, run_param);

    assert(isequal(size(whole,1), N_el), 'reference has the wrong shape');
    assert(any(whole(:) ~= 0), 'nothing was computed -- the test is vacuous');
    assert(all(whole(N_el,:) == 0), ...
        'the unweighted element is not zero; the empty-block case is void');

    % --- element axis: bit for bit -------------------------------------
    for elementsPerBlock = [N_el, 4, 2, 1]
        run_param.RFBlockElements = elementsPerBlock * N_int * N;
        run_param.RFSensorChunkElements = [];
        [f, l] = rf_element_blocks(N_el, N_int, N, 'single', false, run_param);

        split = compute_RF(Transducer, sensor_data, sensor_weights, ...
            Grid, run_param);

        assert(isequal(size(split), size(whole)), ...
            'element blocking changed the shape');
        assert(isequal(class(split), class(whole)), ...
            'element blocking changed the class');
        assert(isequal(split, whole), sprintf( ...
            'weights case %d, %d element block(s): the result changed', ...
            weightCase, numel(f)));
        assert(l(end) == N_el);
    end

    % --- sensor axis: within a few eps ---------------------------------
    run_param.RFBlockElements = [];
    scale = max(abs(whole(:)));
    for chunkRows = [N_sensor, 10, 3, 1]
        run_param.RFSensorChunkElements = chunkRows * M;
        [f, ~] = rf_sensor_chunks(N_sensor, M, false, run_param);

        split = compute_RF(Transducer, sensor_data, sensor_weights, ...
            Grid, run_param);

        err = max(abs(double(split(:)) - double(whole(:)))) / double(scale);
        assert(err < 8*eps('single'), sprintf( ...
            'weights case %d, %d sensor chunk(s): error %.3e is too large', ...
            weightCase, numel(f), err));
    end

    % --- both axes at once ---------------------------------------------
    run_param.RFBlockElements = 2 * N_int * N;
    run_param.RFSensorChunkElements = 3 * M;
    split = compute_RF(Transducer, sensor_data, sensor_weights, ...
        Grid, run_param);
    err = max(abs(double(split(:)) - double(whole(:)))) / double(scale);
    assert(err < 8*eps('single'), ...
        sprintf('both axes split: error %.3e is too large', err));
end

% A single integration point per element takes the scalar-weight branch
% through 1/N_int == 1 and the reshape through a singleton dimension.
Transducer.integration_points = randn(N_el, 1, 3);
Transducer.integration_receive_apodization = rand(N_el, 1);
Transducer.integration_receive_delays = 1e-8 * rand(N_el, 1);
Transducer.integration_weights = 1;
sensor_weights_1 = sensor_weights(1:N_el, :);

run_param.RFBlockElements = [];
run_param.RFSensorChunkElements = [];
whole1 = compute_RF(Transducer, sensor_data, sensor_weights_1, Grid, run_param);

N1 = rf_signal_length(Transducer, Grid, M, 'single');
run_param.RFBlockElements = 2 * 1 * N1;
split1 = compute_RF(Transducer, sensor_data, sensor_weights_1, Grid, run_param);
assert(isequal(split1, whole1), 'N_int == 1: element blocking changed the result');

% A sensor_weights whose rows do not match N_el*N_int is a reshape waiting to
% go wrong silently, so it must be an error.
threw = false;
try
    compute_RF(Transducer, sensor_data, sensor_weights, Grid, run_param);
catch ME
    threw = strcmp(ME.identifier, 'PROTEUS:compute_RF:weightRows');
end
assert(threw, 'a mismatched sensor_weights did not raise weightRows');

disp('test_compute_RF_blocking: all assertions passed');
