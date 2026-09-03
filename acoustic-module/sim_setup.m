function run_param = sim_setup(SimulationParameters)

PATHS = path_setup();

switch SimulationParameters.Solver
    
    case '3D'
        run_param.solver       = 'kspaceFirstOrder3D';
        run_param.DATA_CAST    = 'single'; % run locally and record movie
        run_param.record_movie = true;
    case 'MATLAB'
        % Pure MATLAB 3D solver (no C binary); for testing when 3DC fails
        run_param.solver       = 'kspaceFirstOrder3D';
        run_param.DATA_CAST    = 'single';
        run_param.record_movie = false;
    case '3DC'
        run_param.solver       = 'kspaceFirstOrder3DC';
        run_param.DATA_CAST    = 'single';
        run_param.record_movie = false;
        run_param.DATA_PATH    = PATHS.DataPath;
        run_param.BINARY_PATH  = PATHS.BinaryPath;
    case '3DG'
        run_param.solver       = 'kspaceFirstOrder3DG';
        run_param.DATA_CAST    = 'gpuArray-single';
        run_param.record_movie = false;
        run_param.DEVICE_NUM   = SimulationParameters.DeviceNumber;
        run_param.DATA_PATH    = PATHS.DataPath;
        run_param.BINARY_PATH  = PATHS.BinaryPath;
end

% Data cast for the final RF computation step:
run_param.DATA_CAST_RF = run_param.DATA_CAST;

% Per-stage timings inside the propagation loop. gpuArray work is queued
% asynchronously, so timing a GPU stage means synchronising the device
% first, and that serialises a loop which is otherwise free to overlap.
% Diagnostic only, off unless a run asks for it.
run_param.ProfilePropagation = false;
if isfield(SimulationParameters, 'ProfilePropagation')
    run_param.ProfilePropagation = ...
        logical(SimulationParameters.ProfilePropagation);
end

% Truncation radius of the band-limited delta function used to place the
% microbubbles, i.e. the (2*th+1)^3 stencil each off-grid bubble occupies.
% It is the knob that sizes the recorded transmit: the union of those
% stencils over an acquisition is what the k-Wave sensor records, and with
% tiling that union is the largest array in the run. 4 is the value this
% was fixed at until 2026-09-03, so it stays the default.
%
% The transducer sensor keeps the default whatever this is set to. Its
% record is not what grows with tiling, and its accuracy is a separate
% question from the microbubbles'.
run_param.MicrobubbleDeltaTruncation = 4;
if isfield(SimulationParameters, 'MicrobubbleDeltaTruncation') && ...
        ~isempty(SimulationParameters.MicrobubbleDeltaTruncation)
    th = SimulationParameters.MicrobubbleDeltaTruncation;
    if ~isnumeric(th) || ~isscalar(th) || ~isfinite(th) || ...
            th < 1 || floor(th) ~= th
        error('sim_setup:InvalidDeltaTruncation', ...
            ['SimulationParameters.MicrobubbleDeltaTruncation must be a ' ...
             'positive integer (got %s).'], mat2str(th));
    end
    run_param.MicrobubbleDeltaTruncation = double(th);
end

% Record the transmit once and reuse it across the batch's frames, when the
% acquisition is a single batch. Cheaper by one k-Wave run per pulse, but it
% records the microbubble sensor over the round trip rather than the one-way
% window, which doubles the largest array in the run. Default true, as it
% has been since 2026-08-27.
run_param.CombineTransmitSensors = true;
if isfield(SimulationParameters, 'CombineTransmitSensors') && ...
        ~isempty(SimulationParameters.CombineTransmitSensors)
    run_param.CombineTransmitSensors = ...
        logical(SimulationParameters.CombineTransmitSensors);
end

% Add toolbox paths:
addpath(PATHS.VoxelisationPath);
addpath(PATHS.kWavePath)

% Folder to save the simulation output:
run_param.savedir = PATHS.ResultsPath;

% Folder containing the vessel geometries:
run_param.GeometriesPath = PATHS.GeometriesPath;

% Folder containing the microbubble frames:
run_param.GroundTruthPath = PATHS.GroundTruthPath;

% Folder containing the microbubble simulation module:
run_param.MicrobubblePath = PATHS.MicrobubblePath;

run_param.N_interactions = SimulationParameters.NumberOfInteractions;

% Microbubbles parallel processing properties:
run_param.MicrobubblesBatchSize = 100;
run_param.MicrobubblesUseParfor = 'auto';

end