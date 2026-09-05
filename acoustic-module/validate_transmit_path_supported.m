function validate_transmit_path_supported(run_param, hybrid, num_batches)
%VALIDATE_TRANSMIT_PATH_SUPPORTED Refuse a transmit the solver cannot run.
%
%   The split hybrid path streams its microbubble record out of the binary's
%   own HDF5 output -- run_simulation_to_disk writes it, then
%   project_transmit_to_bubbles reads it in blocks -- because that record is
%   the union of every bubble position in the batch and does not fit host
%   memory. Only the C and CUDA solvers write such a file: sim_setup gives
%   '3DC' and '3DG' a BINARY_PATH and gives '3D' and 'MATLAB' none, since
%   those return their record in memory from MATLAB itself.
%
%   The combined path needs no file -- run_simulation hands the record back in
%   memory -- so a solver without a binary can still run a single-batch
%   combined transmit, and only that.
%
%   Checked here rather than at the transmit, which is reached after the
%   medium, the union mask and the projection pass: ~25 minutes at v11's scale
%   before a run that was never going to work says so. Nothing about the
%   answer needs any of that.

if ~hybrid
    return
end

if isfield(run_param, 'BINARY_PATH') && ~isempty(run_param.BINARY_PATH)
    return
end

if num_batches == 1 && run_param.CombineTransmitSensors
    return
end

if num_batches > 1
    why = sprintf('the acquisition is %d transmit batches', num_batches);
else
    why = 'CombineTransmitSensors is false';
end

error('main_RF:TransmitPathUnsupported', ...
    ['Solver %s writes no k-Wave output file, and %s, so the transmit ' ...
     'would take the split path -- which streams its record out of that ' ...
     'file.\nWhat moves it: SimulationParameters.Solver (3DC and 3DG have ' ...
     'a binary), SimulationParameters.TransmitBatchSize (0 makes the ' ...
     'acquisition a single batch), SimulationParameters.' ...
     'CombineTransmitSensors (true records both sensors in one run and ' ...
     'needs no file).'], ...
    solver_name(run_param), why);

end


function name = solver_name(run_param)
name = '(unnamed)';
if isfield(run_param, 'solver') && ~isempty(run_param.solver)
    name = run_param.solver;
end
end
