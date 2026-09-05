% A transmit path the chosen solver cannot run has to be refused up front.
%
% The split hybrid path streams its microbubble record out of the binary's own
% HDF5 output, because that record is the union of every bubble position in the
% batch and does not fit host memory. Only '3DC' and '3DG' get a BINARY_PATH
% from sim_setup; '3D' and 'MATLAB' return their record in memory from MATLAB
% itself and write no such file. So those two can run a single-batch combined
% transmit and nothing else -- and used to find that out ~25 minutes in, after
% the medium, the union mask and the projection pass.

clear rpb okb eb

%% A solver with a binary runs any combination.
rpb = struct('solver', 'kspaceFirstOrder3DG', 'BINARY_PATH', '/opt/kwave', ...
             'CombineTransmitSensors', false);
validate_transmit_path_supported(rpb, true, 5);
validate_transmit_path_supported(rpb, true, 1);

%% A non-hybrid run never takes either path.
rpb = struct('solver', 'kspaceFirstOrder3D', 'CombineTransmitSensors', false);
validate_transmit_path_supported(rpb, false, 5);

%% Without a binary, a single-batch combined transmit is still fine.
% run_simulation hands that record back in memory; no file is involved.
rpb = struct('solver', 'kspaceFirstOrder3D', 'CombineTransmitSensors', true);
validate_transmit_path_supported(rpb, true, 1);

%% Without a binary, turning the combined sensor off is refused.
rpb = struct('solver', 'kspaceFirstOrder3D', 'CombineTransmitSensors', false);
okb = false;
try
    validate_transmit_path_supported(rpb, true, 1);
    okb = true;
catch eb
    assert(strcmp(eb.identifier, 'main_RF:TransmitPathUnsupported'), ...
        'wrong error: %s', eb.identifier)
    % The message has to name what moves it, or it only says no.
    for k = {'Solver', 'TransmitBatchSize', 'CombineTransmitSensors'}
        assert(contains(eb.message, k{1}), 'message does not name %s', k{1})
    end
    assert(contains(eb.message, 'CombineTransmitSensors is false'), ...
        'the message must say which of the two conditions refused it')
end
assert(~okb, 'a split transmit without a binary must not be attempted')

%% Without a binary, more than one transmit batch is refused.
rpb = struct('solver', 'kspaceFirstOrder3D', 'CombineTransmitSensors', true);
okb = false;
try
    validate_transmit_path_supported(rpb, true, 3);
    okb = true;
catch eb
    assert(strcmp(eb.identifier, 'main_RF:TransmitPathUnsupported'))
    assert(contains(eb.message, '3 transmit batches'), ...
        'the message must say how many batches refused it')
end
assert(~okb, 'multiple batches force the split path, which needs a binary')

disp('test_validate_transmit_path_supported: all assertions passed')
