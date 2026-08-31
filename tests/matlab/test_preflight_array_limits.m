% The preflight must pass the configuration that blocking can hold, and
% refuse the one it cannot -- before the transmit either way.
%
% v10 is the case that motivated it: three of its receive arrays are over the
% gpuArray element limit whole, and all of them fit once blocked, so the run
% must be allowed to start. The refusal case is the one no code change fixes:
% a single element's own work over the limit, where the grid, the element
% height or IntegrationDensity has to give.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));
run_log('reset');

LIMIT = double(intmax('int32'));

% Built from scratch, not assigned into: scripts run through run() share the
% base workspace, and a stray RFBlockElements from another test would change
% the very partition this one checks.
Grid = struct('dt', 6.8033e-9);
run_param = struct('DATA_CAST_RF', 'gpuArray-single');

%% v10 production geometry: over the limit whole, fine once blocked.
N_el = 192; N_int = 990; N_sensor = 201795; M = 12444;
Transducer = struct();
Transducer.integration_points = zeros(N_el, N_int, 3);
Transducer.integration_receive_delays = 1e-8 * ones(N_el, N_int);

assert(N_el*N_int*M > LIMIT, 'v10 must be over the limit whole');
preflight_array_limits(Transducer, N_sensor, M, 3, Grid, run_param);

% And on the host retry, where there is no device limit to trip at all.
run_param_cpu = run_param;
run_param_cpu.DATA_CAST_RF = 'single';
preflight_array_limits(Transducer, N_sensor, M, 3, Grid, run_param_cpu);

%% v7 production geometry: under the limit, nothing to say.
preflight_array_limits(Transducer, 113963, 9339, 3, Grid, run_param);

%% One element whose own work is over the limit: no blocking helps.
Pathological = struct();
Pathological.integration_points = zeros(1, 200000, 3);
Pathological.integration_receive_delays = zeros(1, 200000);

threw = '';
try
    preflight_array_limits(Pathological, 1000, 20000, 1, Grid, run_param);
catch ME
    threw = ME.identifier;
end
assert(strcmp(threw, 'PROTEUS:preflight:arrayOverLimit'), ...
    'a configuration that cannot be blocked was allowed to start');

% The same shape on the host is not a device failure, so it must not error:
% host arrays have no element cap, only memory the preflight reports.
preflight_array_limits(Pathological, 1000, 20000, 1, Grid, run_param_cpu);

disp('test_preflight_array_limits: all assertions passed');
