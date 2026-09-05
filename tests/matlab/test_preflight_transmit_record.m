% The transmit record has to be sized before the transmit runs, and the path
% has to be chosen on sizes rather than on a setting alone.
%
% preflight_array_limits sizes the receive path, all of it transducer-shaped.
% Nothing sized the transmit, and the transmit is what failed first: the v11
% run of 2026-09-03 asked the binary for a 550 GB output file on a ~145 GB
% disk and found out 36 minutes in, at 25% of the first pulse.
%
% Sizing the disk was not enough. The combined path reads its whole record
% into host memory (run_simulation ends in h5read) and then takes two subsets
% off it, so its peak is ~1.5x the record where the split path's is a few
% percent of it -- and the combined path is the default. A record between the
% host's memory and the disk's free space passed this check and died hours
% later out of memory.

clear rpz Pz okz combz errz

% One pulse, 40 bubbles in each of 10 frames, over a 1e6-point mask. Sized
% from the bubble counts rather than a built projection, so the check runs
% between the union mask and the projection pass rather than after it.
Pz = repmat(40, 10, 1);

% At this shape:
%   split    record 37.3 GB   host peak  14.9 GB
%   combined record 89.4 GB   host peak 141.6 GB
NMB = 1e6; NTR = 2e5; TMB = 10000; TTR = 20000;

% BINARY_PATH is what makes the fallback possible at all: the split path
% streams its record out of the binary's own output file. Without it there is
% nothing to fall back to -- see the last case.
rpz = struct('DATA_PATH', tempdir, ...
             'BINARY_PATH', '/opt/kwave', ...
             'MicrobubbleDeltaTruncation', 4, ...
             'CombineTransmitSensors', false);

%% A split record that fits both budgets runs, and stays split.
rpz.FreeDiskBytes   = 100 * 2^30;
rpz.HostMemoryBytes = 100 * 2^30;
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, false, Pz, rpz);
assert(combz == false, 'a split request must come back split');

%% A record larger than the disk on both paths must not start.
rpz.FreeDiskBytes = 30 * 2^30;      % split needs 37.3 * 1.1 = 41 GB
okz = false;
try
    preflight_transmit_record(NMB, NTR, TMB, TTR, false, Pz, rpz);
    okz = true;
catch e
    assert(strcmp(e.identifier, 'preflight_transmit_record:RecordTooLarge'), ...
        'wrong error: %s', e.identifier)
    % The message has to name the knobs that move it, or it only says no.
    for k = {'MicrobubbleDeltaTruncation', 'CombineTransmitSensors', ...
             'TransmitBatchSize', 'Tiling.NumTiles'}
        assert(contains(e.message, k{1}), 'message does not name %s', k{1})
    end
end
assert(~okz, 'a record larger than the disk must not be attempted')

%% Combined is kept when both budgets hold it.
rpz.CombineTransmitSensors = true;
rpz.FreeDiskBytes   = 200 * 2^30;   % combined needs 89.4 * 1.1 = 98.3 GB
rpz.HostMemoryBytes = 200 * 2^30;   % combined peak 141.6 GB
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
assert(combz == true, 'combined fits both budgets and must be kept');

%% Combined that fits the disk but not host memory falls back to split.
% This is the hole the disk check left: 141.6 GB of peak against a 100 GB
% host, on a disk with room to spare. It used to pass and die hours later.
rpz.FreeDiskBytes   = 200 * 2^30;
rpz.HostMemoryBytes = 100 * 2^30;
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
assert(combz == false, ...
    'combined over the memory budget must fall back, not proceed');

%% Combined that outgrows the disk falls back rather than refusing the run.
% The split record is half the window and fits here, so there is a run to be
% had. This is the v11 case: it errored, where falling back would have run.
rpz.FreeDiskBytes   = 50 * 2^30;    % combined 98.3 GB no, split 41 GB yes
rpz.HostMemoryBytes = 200 * 2^30;
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
assert(combz == false, 'combined over the disk budget must fall back');

%% When neither path fits, it is still an error and not a fallback.
rpz.FreeDiskBytes   = 30 * 2^30;
rpz.HostMemoryBytes = 200 * 2^30;
okz = false;
try
    preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
    okz = true;
catch e
    assert(strcmp(e.identifier, 'preflight_transmit_record:RecordTooLarge'))
end
assert(~okz, 'neither path fits: the run must be refused')

%% An unknown budget reports rather than blocks.
% MATLAB cannot always ask the filesystem or the cgroup, and a run that would
% be fine must not be refused for that.
clear rpz
rpz = struct('MicrobubbleDeltaTruncation', 4, 'CombineTransmitSensors', false);
preflight_transmit_record(NMB, NTR, TMB, TTR, false, Pz, rpz);

%% Without a binary there is no fallback, so the path is left alone.
% The split path streams its record out of the binary's output file. A solver
% that writes no such file can only run the combined transmit, over budget or
% not -- validate_transmit_path_supported has already refused the cases where
% the split path would have been mandatory. Downgrading here would hand
% run_simulation_to_disk a run it cannot do.
clear rpz
rpz = struct('DATA_PATH', tempdir, 'MicrobubbleDeltaTruncation', 4, ...
             'CombineTransmitSensors', true, ...
             'FreeDiskBytes', 200 * 2^30, 'HostMemoryBytes', 100 * 2^30);
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
assert(combz == true, ...
    'with no binary to stream with, the combined path must be kept');

%% An unknown memory budget leaves a combined request alone.
% NaN is how a caller says "I cannot tell you": reported, never enforced.
rpz.FreeDiskBytes   = 200 * 2^30;
rpz.HostMemoryBytes = NaN;
rpz.CombineTransmitSensors = true;
combz = preflight_transmit_record(NMB, NTR, TMB, TTR, true, Pz, rpz);
assert(combz == true, ...
    'an unknown memory budget must not silently downgrade the path');

disp('test_preflight_transmit_record: all assertions passed')
