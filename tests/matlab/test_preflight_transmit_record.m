% The transmit record has to be sized before the transmit runs.
%
% preflight_array_limits sizes the receive path, all of it transducer-shaped.
% Nothing sized the transmit, and the transmit is what failed first: the v11
% run of 2026-09-03 asked the binary for a 550 GB output file on a ~145 GB
% disk and found out 36 minutes in, at 25% of the first pulse.

clear rpz Wz Pz okz errz

rpz = struct('DATA_PATH', tempdir, ...
             'MicrobubbleDeltaTruncation', 4, ...
             'CombineTransmitSensors', false);

% One pulse, 40 bubbles in each of 10 frames, over a 1e6-point mask. Sized
% from the bubble counts rather than a built projection, so the check runs
% between the union mask and the projection pass rather than after it.
Pz = repmat(40, 10, 1);

% 1e6 x 10000 x 4 bytes = 40 GB, and 1.1x of that against 100 GB free.
rpz.FreeDiskBytes = 100 * 2^30;
preflight_transmit_record(1e6, 2e5, 10000, 20000, false, Pz, rpz);
disp('  (fits: no error, as expected)')

% The same record against 30 GB must not start.
rpz.FreeDiskBytes = 30 * 2^30;
okz = false;
try
    preflight_transmit_record(1e6, 2e5, 10000, 20000, false, Pz, rpz);
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

% The combined path is sized on the round trip, so it can fail where the
% split path fits -- which is the whole reason it became a setting.
rpz.FreeDiskBytes = 50 * 2^30;
preflight_transmit_record(1e6, 2e5, 10000, 20000, false, Pz, rpz);
okz = false;
try
    rpz.CombineTransmitSensors = true;
    preflight_transmit_record(1e6, 2e5, 10000, 20000, true, Pz, rpz);
    okz = true;
catch e
    assert(strcmp(e.identifier, 'preflight_transmit_record:RecordTooLarge'))
end
assert(~okz, 'combined is twice the window and must be sized as such')

% An unknown free space reports rather than blocks: MATLAB on Linux cannot
% always ask, and a run that would be fine must not be refused for that.
clear rpz
rpz = struct('MicrobubbleDeltaTruncation', 4, 'CombineTransmitSensors', false);
preflight_transmit_record(1e6, 2e5, 10000, 20000, false, Pz, rpz);

disp('test_preflight_transmit_record: all assertions passed')
