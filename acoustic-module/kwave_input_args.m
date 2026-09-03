function input_args = kwave_input_args(run_param)
%KWAVE_INPUT_ARGS The k-Wave options every simulation in this module shares.
%
%   Split out so run_simulation and run_simulation_to_disk cannot drift
%   apart: the two paths must set up the same grid, PML and precision, or
%   the streamed transmit would not be the same simulation as the cached
%   one.

PML = run_param.PML;

input_args = {...
    'PMLInside', false,...
    'PMLAlpha',  PML.Alpha, ...
    'PMLSize',   [PML.X_SIZE, PML.Y_SIZE, PML.Z_SIZE],...
    'DataCast',  run_param.DATA_CAST, ...
    'Smooth',    false};

end
