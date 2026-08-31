function main_RF(settingsfile, groundtruthfolder, savefolder, varargin)
% =========================================================================
% SIMULATE RF DATA
% input: settingsfile:      the file containing the simulation settings
%        groundtruthfolder: the folder containing the ground truth data
%        savefolder:        the folder to save the RF data
%        varargin{1}:       continue with the same k-Wave medium (boolean)
%        varargin{2}:       frame number to continue from (integer)
%        varargin{3}:       frame number to stop after (integer)
%        varargin{4}:       GPU device number (counting from zero)
%
% Alina Kuliesh,  Delft University of Technology
% Nathan Blanken, University of Twente
% 2022
% =========================================================================

% Get full paths and add modules to MATLAB path:
[settingsfile, groundtruthfolder, savedir] = ...
    sim_startup(settingsfile, groundtruthfolder, savefolder);

load(settingsfile,'Acquisition','Geometry','Medium',...
    'Microbubble','SimulationParameters', 'Transducer', 'Transmit')

% Process optional input arguments:
[Acquisition, SimulationParameters] = ...
    input_handling(Acquisition, SimulationParameters, varargin);

% simulation settings
run_param = sim_setup(SimulationParameters);

% Microbubble parallel processing properties:
Microbubble.BatchSize = run_param.MicrobubblesBatchSize;
Microbubble.UseParfor = run_param.MicrobubblesUseParfor;

% Properties of the representation of the transducer on the grid:
if isfield(SimulationParameters,'TransducerOnGrid')
    Transducer.OnGrid = SimulationParameters.TransducerOnGrid;
else
    Transducer.OnGrid = false;
end
if isfield(SimulationParameters,'IntegrationDensity')
    Transducer.IntegrationDensity = ...
        SimulationParameters.IntegrationDensity;
else
    Transducer.IntegrationDensity = 1;
end

% Location of the geometry data:
Geometry.GeometriesPath = run_param.GeometriesPath;

estimate = false;   % Estimate time and memory consumption
if ~isfield(Medium,'Save')
    Medium.Save = true; % Save the k-Wave medium
end

% Check if the microbubble properties and the acquisition properties for
% the ground truth data match the simulation properties:
check_ground_truth_data(groundtruthfolder,Acquisition,Microbubble,savedir)
flow_params = load(fullfile(groundtruthfolder, 'FlowSimulationParameters.mat'), ...
    'FlowSimulationParameters');
FlowSimulationParameters = flow_params.FlowSimulationParameters;

disp(['RF data will be saved in: ' newline savedir '.' newline])

% Record what produced this run -- simulator commit, effective settings, and
% environment -- beside the frames themselves, before any of them exist. A
% manifest written at the end is lost for exactly the runs whose provenance is
% hardest to reconstruct: the ones that crash or are cut short.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'scripts'))
Segment = struct('StartFrame', Acquisition.StartFrame, ...
    'EndFrame', Acquisition.EndFrame);
write_run_manifest(savedir, settingsfile, Segment);


% Define the k-Wave grid:
disp('Creating k-Wave grid ...')
[kgrid, Grid] = define_grid(SimulationParameters, Geometry);

% Define the k-Wave medium:
if Acquisition.Continue
    disp('Loading k-Wave medium ...')
    medium_file = [savedir '/medium.mat'];
    medium_data = load(medium_file,'medium');
    medium = medium_data.medium;
    medium_var_info = whos('-file', medium_file);
    medium_vars = {medium_var_info.name};
    if any(strcmp(medium_vars, 'medium_metadata'))
        medium_metadata_data = load(medium_file, 'medium_metadata');
        medium_metadata = medium_metadata_data.medium_metadata;
    else
        medium_metadata = struct();
    end
    Medium.Save = false; % No need to save the medium again
else
    disp('Creating k-Wave medium ...')
    [medium, vessel_grid, medium_metadata] = ...
        define_medium(Grid, Medium, Geometry, FlowSimulationParameters);
end
assert_tiling_metadata_matches(FlowSimulationParameters, medium_metadata);

% Save the k-Wave medium
if Medium.Save
    disp('Saving k-Wave medium ...')
    save([savedir '/medium.mat'],'medium','vessel_grid','Grid', ...
        'medium_metadata','-v7.3')
end

% Distribute integration points at the transducer surface:
Transducer = get_transducer_integration_points(Transducer, Grid);
Transducer = get_transducer_integration_delays(Transducer, Medium);

% record signals long enough for back and forth pass of the wave
run_param = compute_travel_times(run_param, ...
    Geometry,Medium,Transducer,Transmit);

run_param.PML = Grid.PML;

% create the time array
kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;

% Filter and resample transmit signal:
Transmit = preprocess_transmit(Transmit,Medium,kgrid);

% Acquisition sequence
switch Acquisition.PulsingScheme
    case 'Amplitude modulation'
        sequence = {'odd' 'even' 'all'};
    case 'Pulse inversion'
        sequence = {'plus' 'minus'};
    case 'Standard'
        sequence = {'pulse'};
    case 'Amplitude modulation with pulse inversion'
        sequence = {'odd' 'even' 'minus'};
end

%==========================================================================
% strucutre for time and memory estimation
param.c_max = Medium.SpeedOfSoundMaximum;
param.CFL = SimulationParameters.CFL;
param.tr = run_param.tr;
param.num_frames = Acquisition.EndFrame - Acquisition.StartFrame + 1;
param.num_pulse = Acquisition.NumberOfPulses;
param.num_int = SimulationParameters.NumberOfInteractions;
param.max_mb = Microbubble.Number;
param.PML = Grid.PML;

%==========================================================================
% First iteration: transducer send pulse; MBs record pulse

disp('Creating k-Wave sensor object for transducer.')
[sensor_transducer, sensor_weights] = define_sensor_transducer(...
    Transducer, Grid);

mask_idx_trans = find(logical(sensor_transducer.mask));

source_transducer = cell(1,length(sequence));

% define_source_transducer is fast; always recompute. Only the run_simulation
% call below (the actual transmit pulse propagation) is expensive enough to
% cache.
for pulse_seq_idx = 1 : length(sequence)
    Transmit.SeqPulse = sequence{pulse_seq_idx};
    disp('Creating k-Wave source object for transducer.')
    source_transducer{pulse_seq_idx} = define_source_transducer(...
        Transducer, Transmit, Medium, Grid, transpose(sensor_weights), ...
        mask_idx_trans);
end

%==========================================================================
% Second & Third iterations (frames + MB part)

% Add the microbubble module to the path for the whole acquisition. This used
% to be done and undone inside the per-frame simulator call, which changed the
% search path twice per frame; MATLAB drops functions from memory when the
% path changes, so every frame re-parsed the module and reset the run log's
% banner state. The cleanup object restores the path on any exit, including
% the early return of the pressure-capture path and any error.
addedMicrobubblePaths = add_microbubble_path(run_param);
microbubblePathCleanup = onCleanup(@() ...
    remove_microbubble_path(addedMicrobubblePaths)); %#ok<NASGU>

% Timer for frames+MB part (after initial transmit):
run_log('reset');
t_frames_start = tic;
tstart = tic;
execution_times = zeros(1,Acquisition.NumberOfFrames);
saveExecutionTimes = false;

num_frames_to_process = Acquisition.EndFrame - Acquisition.StartFrame + 1;

if SimulationParameters.HybridSimulation
    transmit_batch_size = get_transmit_batch_size(...
        SimulationParameters, Acquisition);
    frame_batches = make_frame_batches(Acquisition.StartFrame, Acquisition.EndFrame, transmit_batch_size);
    num_batches = size(frame_batches, 1);

    % The transducer-only and MB-only runs propagate the same pulse through
    % the same medium; they differ only in which points are recorded and for
    % how long. The split pays off only because the MB run is short and the
    % transducer run is done once: it costs one round-trip run plus one
    % one-way run per batch, against one round-trip run per batch for a
    % combined sensor. Measured on run_20260827_082616: 816.58 s round trip,
    % 449.50 s one way, so the split wins from three batches on and loses
    % 449.50 s per pulse at a single batch - which is the production setting.
    % So record both sensor sets in one run whenever there is one batch.
    combine_transmit_sensors = num_batches == 1;

    n_transducer_time = floor(run_param.tr(3) / kgrid.dt) + 1;
    n_mb_time = floor(run_param.tr(1) / kgrid.dt) + 1;

    % Simulation time and memory estimation:
    if estimate == true
        beta_coeff_file = ['time-estimation' filesep 'beta_coeff.mat'];
        estim_time_mem(Grid, source_transducer{1}, param, beta_coeff_file);
    end

    sensor_data_transducer_1iter = cell(1,length(sequence));

    if ~combine_transmit_sensors
        kgrid.Nt = n_transducer_time;
        for pulse_seq_idx = 1 : length(sequence)
            disp('Simulating transducer-only transmit wave.')
            t_tx = tic;
            sensor_data_transducer_1iter{pulse_seq_idx} = run_simulation(...
                run_param, kgrid, medium, source_transducer{pulse_seq_idx}, ...
                sensor_transducer);
            fprintf('[TIMING] Transducer transmit wave (pulse %d): %.2f s\n', ...
                pulse_seq_idx, toc(t_tx));
        end
    end

    for batch_idx = 1:num_batches
        batch_start = frame_batches(batch_idx, 1);
        batch_end = frame_batches(batch_idx, 2);
        fprintf('=== MB transmit batch %d/%d: frames %d-%d ===\n', ...
            batch_idx, num_batches, batch_start, batch_end);

        AcquisitionBatch = Acquisition;
        AcquisitionBatch.StartFrame = batch_start;
        AcquisitionBatch.EndFrame = batch_end;
        [sensor_MB_batch, MB_idx_all, max_mb_batch, bubble_counts] = ...
            define_sensor_MB_all(Grid, groundtruthfolder, AcquisitionBatch, ...
            length(sequence), Geometry);
        param.max_mb = max(param.max_mb, max_mb_batch);
        if batch_idx == 1
            validate_evaluation_capture_yield_if_requested(...
                SimulationParameters, bubble_counts);
        end

        if ~isempty(intersect(MB_idx_all, mask_idx_trans))
            warning('Microbubbles on transducer.')
        end

        sensor_data_MB_1iter = cell(1,length(sequence));

        if combine_transmit_sensors
            % One sensor over both masks, recorded for the round trip the
            % transducer needs. A grid point carried by both masks is stored
            % once and read by both extractions below.
            sensor_combined.mask = logical(...
                sensor_MB_batch.mask + sensor_transducer.mask);
            sensor_combined.record = sensor_MB_batch.record;
            mask_idx_combined = find(sensor_combined.mask);
            mask_idx_MB_batch = find(logical(sensor_MB_batch.mask));
            kgrid.Nt = n_transducer_time;

            for pulse_seq_idx = 1 : length(sequence)
                disp('Simulating combined transducer and MB transmit wave.')
                t_tx = tic;
                sensor_data_combined = run_simulation(...
                    run_param, kgrid, medium, ...
                    source_transducer{pulse_seq_idx}, sensor_combined);
                fprintf('[TIMING] Combined transmit wave (pulse %d, frames %d-%d): %.2f s\n', ...
                    pulse_seq_idx, batch_start, batch_end, toc(t_tx));

                % Split the one run into the two sensor sets the split path
                % produced separately. The bubbles only ever read the one-way
                % window, so what is held through the frame loop is the same
                % size the split path held.
                sensor_data_transducer_1iter{pulse_seq_idx} = ...
                    extract_sensor_subset(sensor_data_combined, ...
                    mask_idx_combined, mask_idx_trans, n_transducer_time);
                sensor_data_MB_1iter{pulse_seq_idx} = ...
                    extract_sensor_subset(sensor_data_combined, ...
                    mask_idx_combined, mask_idx_MB_batch, n_mb_time);
                clear sensor_data_combined
            end
        else
            mask_idx_MB_batch = find(logical(sensor_MB_batch.mask));
            kgrid.Nt = n_mb_time;

            for pulse_seq_idx = 1 : length(sequence)
                disp('Simulating MB-only transmit wave.')
                t_tx = tic;
                sensor_data_MB_1iter{pulse_seq_idx} = run_simulation(...
                    run_param, kgrid, medium, ...
                    source_transducer{pulse_seq_idx}, sensor_MB_batch);
                fprintf('[TIMING] MB transmit wave (pulse %d, frames %d-%d): %.2f s\n', ...
                    pulse_seq_idx, batch_start, batch_end, toc(t_tx));
            end
        end

        for frame = batch_start : batch_end
            t_frame = tic;

            RF = cell(1,length(sequence));
            Frame = cell(1,length(sequence));

            for pulse_seq_idx = 1 : length(sequence)

                % Reading this frame's bubbles and building its sensor.
                t_load = tic;
                MB = load_microbubbles(groundtruthfolder, frame, pulse_seq_idx, ...
                    Geometry, Acquisition.NumberOfFrames);

                % define the sensor of the current frame
                [sensor_frame, sensor_weights_frame, MB, run_param.max_dist] = ...
                    define_sensor_MB(Grid, MB);

                mask_idx_frame = find(logical(sensor_frame.mask));
                run_log('stage', 'load', toc(t_load));

                % Taking the batch's recorded transmit down to this frame's
                % bubble positions. The batch sensor covers every frame in
                % the batch, so this both selects rows and applies the
                % per-bubble interpolation weights.
                % Split because the two halves have different fixes. The
                % selection re-intersects mask_idx_MB_batch -- constant for
                % the whole batch -- against this frame's mask on every frame,
                % so it is a hoist if it dominates. The weighting is a sparse
                % product that has to go through double, since MATLAB has no
                % single-precision sparse, so it is not. Measured together at
                % 2.4 s of a 14.8 s frame, which is 16% with no way to tell
                % which half.
                t_sense = tic;
                t_idx = tic;
                sensor_data_MB = extract_sensor_subset(...
                    sensor_data_MB_1iter{pulse_seq_idx}, ...
                    mask_idx_MB_batch, mask_idx_frame, n_mb_time);
                sensor_data_trans = sensor_data_transducer_1iter{pulse_seq_idx};
                sync_if_on_device(sensor_data_MB.p);
                run_log('stage', 'idx', toc(t_idx));

                % Pressure sensed by the microbubbles
                t_weights = tic;
                sensed_p = sensor_weights_frame*double(sensor_data_MB.p);
                sensed_p = cast(full(sensed_p),class(sensor_data_MB.p));
                sync_if_on_device(sensed_p);
                run_log('stage', 'weights', toc(t_weights));

                run_log('stage', 'sense', toc(t_sense));

                stopAfterCapture = capture_sensed_pressure_if_requested(...
                    SimulationParameters, sensed_p, MB, kgrid, Medium, ...
                    Microbubble, Transmit, frame, pulse_seq_idx, ...
                    Acquisition.StartFrame);
                if stopAfterCapture
                    return
                end

                % Complete the transducer sensor data with microbubble sources:
                try
                    sensor_data = hybrid_simulator(...
                        mask_idx_trans,...
                        sensed_p, ...
                        MB, Grid, medium, run_param, ...
                        Medium, Microbubble, Transmit);
                catch exception
                    rethrow(bubble_solver_context(...
                        exception, frame, pulse_seq_idx))
                end

                % Update sensor data transducer. Column-blocked and in
                % place: both operands are the full [N_sensor x Nt] record,
                % 9.4 GB each at v10's grid, and writing the sum into a
                % third array put the host peak at three copies of it.
                accum_cols = max(1, ...
                    floor(2^26 / size(sensor_data.p,1)));
                for col_start = 1:accum_cols:size(sensor_data.p,2)
                    cols = col_start : min(col_start + accum_cols - 1, ...
                        size(sensor_data.p,2));
                    sensor_data.p(:,cols) = sensor_data.p(:,cols) + ...
                        sensor_data_trans.p(:,cols);
                end

                % Compute element RF data recorded by transducer:
                t_rf = tic;
                [RF{pulse_seq_idx}, run_param] = compute_RF_data(...
                    Transducer,sensor_data,sensor_weights,Grid,run_param);
                run_log('stage', 'RF', toc(t_rf));

                Frame{pulse_seq_idx} = MB;

            end

            % Save data
            dt = kgrid.dt;
            % Find out how many zero padding you'll need for file name
            num_padding=num2str(length(num2str(Acquisition.NumberOfFrames)));
            file_name = ['Frame_', num2str(frame,['%0',num_padding,'i']),'.mat'];
            t_save = tic;
            save([savedir filesep file_name], 'RF', 'dt', 'Frame')
            run_log('stage', 'save', toc(t_save));

            execution_times(frame) = toc(tstart);
            run_log('frame', frame, Acquisition.EndFrame, toc(t_frame));

        end
    end
else
    [sensor_MB_all, MB_idx_all, max_mb, bubble_counts] = ...
        define_sensor_MB_all(Grid, groundtruthfolder, Acquisition, ...
        length(sequence), Geometry);
    param.max_mb = max_mb;
    validate_evaluation_capture_yield_if_requested(...
        SimulationParameters, bubble_counts);

    if ~isempty(intersect(MB_idx_all, mask_idx_trans))
        warning('Microbubbles on transducer.')
    end

    sensor = sensor_MB_all;
    kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;
    sensor_data_1iter = cell(1,length(sequence));

    for pulse_seq_idx = 1 : length(sequence)
        % Simulation time and memory estimation:
        if pulse_seq_idx == 1 && estimate == true
            beta_coeff_file = ['time-estimation' filesep 'beta_coeff.mat'];
            estim_time_mem(Grid, source_transducer{pulse_seq_idx}, param, ...
                beta_coeff_file);
        end

        disp('Simulating transmit wave.')
        t_tx = tic;
        sensor_data_1iter{pulse_seq_idx} = run_simulation(run_param, kgrid, ...
            medium, source_transducer{pulse_seq_idx}, sensor);
        fprintf('[TIMING] Transmit wave (pulse %d): %.2f s\n', ...
            pulse_seq_idx, toc(t_tx));
    end

    for frame = Acquisition.StartFrame : Acquisition.EndFrame
        t_frame = tic;

        RF = cell(1,length(sequence));
        Frame = cell(1,length(sequence));

        for pulse_seq_idx = 1 : length(sequence)

            MB = load_microbubbles(groundtruthfolder, frame, pulse_seq_idx, Geometry, ...
                Acquisition.NumberOfFrames);

            % define the sensor of the current frame
            [sensor_frame, sensor_weights_frame, MB, run_param.max_dist] = ...
                define_sensor_MB(Grid, MB);

            mask_idx       = find(logical(sensor.mask));
            mask_idx_frame = find(logical(sensor_frame.mask));

            % Split sensor data into microbubble sensor data.
            [sensor_data_MB, ~] = extract_sensor_data(...
                sensor_data_1iter{pulse_seq_idx}, ...
                mask_idx, mask_idx_trans, mask_idx_frame, run_param, kgrid);

            % Pressure sensed by the microbubbles
            sensed_p = sensor_weights_frame*double(sensor_data_MB.p);
            sensed_p = cast(full(sensed_p),class(sensor_data_MB.p));

            stopAfterCapture = capture_sensed_pressure_if_requested(...
                SimulationParameters, sensed_p, MB, kgrid, Medium, ...
                Microbubble, Transmit, frame, pulse_seq_idx, ...
                Acquisition.StartFrame);
            if stopAfterCapture
                return
            end

            try
                sensor_data = full_simulator(...
                    source_transducer{pulse_seq_idx}, ...
                    sensor_transducer,...
                    sensor_frame,sensor_weights_frame,mask_idx_frame,...
                    sensed_p,...
                    MB, kgrid, Grid, medium, run_param, ...
                    Medium, Microbubble, Transmit);
            catch exception
                rethrow(bubble_solver_context(exception, frame, pulse_seq_idx))
            end

            % Compute element RF data recorded by transducer:
            t_rf = tic;
            [RF{pulse_seq_idx}, run_param] = compute_RF_data(...
                Transducer,sensor_data,sensor_weights,Grid,run_param);
            run_log('stage', 'RF', toc(t_rf));

            Frame{pulse_seq_idx} = MB;

        end

        % Save data
        dt = kgrid.dt;
        % Find out how many zero padding you'll need for file name
        num_padding=num2str(length(num2str(Acquisition.NumberOfFrames)));
        file_name = ['Frame_', num2str(frame,['%0',num_padding,'i']),'.mat'];
        save([savedir filesep file_name], 'RF', 'dt', 'Frame')

        execution_times(frame) = toc(tstart);
        run_log('frame', frame, Acquisition.EndFrame, toc(t_frame));

    end
end

% Report time for frames + MB part
frames_elapsed = toc(t_frames_start);
run_log('summary', num_frames_to_process, frames_elapsed);

% Save execution times for performance quantification if requested:
if saveExecutionTimes == true
    file_name = 'execution_time_history.mat';
    save([savedir filesep file_name], 'execution_times')
end

end


function addedPaths = add_microbubble_path(run_param)
% Add the microbubble module for the acquisition, reporting what was new.
%
% Only the entries this call actually adds are returned. A caller may already
% have the module on the path and still need it after main_RF returns - the
% GPU solver evaluation adds it, calls main_RF to capture pressure, and then
% replays that pressure through compute_bubble_mass_source. Removing an entry
% this function did not add left that caller without the module.

candidatePaths = {run_param.MicrobubblePath, ...
    fullfile(run_param.MicrobubblePath, 'functions')};
addedPaths = {};
for i = 1:numel(candidatePaths)
    if ~is_on_search_path(candidatePaths{i})
        addpath(candidatePaths{i})
        addedPaths{end + 1} = candidatePaths{i}; %#ok<AGROW>
    end
end

end


function remove_microbubble_path(addedPaths)
% Undo the acquisition-wide addpath of the microbubble module.

for i = 1:numel(addedPaths)
    if is_on_search_path(addedPaths{i})
        rmpath(addedPaths{i})
    end
end

end


function tf = is_on_search_path(candidatePath)
% True when CANDIDATEPATH is already an entry of the MATLAB search path.

entries = strsplit(path, pathsep);
normalize = @(p) regexprep(p, [regexptranslate('escape', filesep) '+$'], '');
tf = any(strcmp(cellfun(normalize, entries, 'UniformOutput', false), ...
    normalize(candidatePath)));

end


function validate_evaluation_capture_yield_if_requested(...
    SimulationParameters, bubbleCounts)
% Reject insufficient first-frame/first-pulse yield before k-Wave runs.

if ~isfield(SimulationParameters, 'EvaluationCapture')
    return
end
config = SimulationParameters.EvaluationCapture;
if ~isfield(config, 'Enabled') || ~isscalar(config.Enabled) || ...
        ~logical(config.Enabled) || ~isfield(config, 'RequestedBubbleCount')
    return
end
requestedBubbleCount = config.RequestedBubbleCount;
if ~isscalar(requestedBubbleCount) || ~isnumeric(requestedBubbleCount) || ...
        requestedBubbleCount < 1 || ...
        requestedBubbleCount ~= floor(requestedBubbleCount)
    error('main_RF:InvalidRequestedBubbleCount', ...
        'EvaluationCapture.RequestedBubbleCount must be a positive integer.');
end
validBubbleCount = bubbleCounts(1, 1);
seededBubbleCount = validBubbleCount;
if isfield(config, 'SeededBubbleCount')
    seededBubbleCount = config.SeededBubbleCount;
end
if validBubbleCount < requestedBubbleCount
    error('main_RF:InsufficientCapturedBubbles', ...
        ['Evaluation capture seeded %d bubble(s), found %d valid in-grid ' ...
         'bubble(s) in the first frame and first pulse, and requires %d.'], ...
        seededBubbleCount, validBubbleCount, requestedBubbleCount);
end
end

function stopAfterCapture = capture_sensed_pressure_if_requested(...
    SimulationParameters, sensed_p, MB, kgrid, Medium, Microbubble, ...
    Transmit, frame, pulseSequenceIndex, firstFrame)
% Save the first incident-pressure input for offline solver evaluation.

stopAfterCapture = false;
if ~isfield(SimulationParameters, 'EvaluationCapture') || ...
        frame ~= firstFrame || pulseSequenceIndex ~= 1
    return
end

config = SimulationParameters.EvaluationCapture;
if ~isfield(config, 'Enabled') || ~isscalar(config.Enabled) || ...
        ~logical(config.Enabled)
    return
end
if ~isfield(config, 'OutputPath') || isempty(config.OutputPath)
    error('main_RF:MissingEvaluationCapturePath', ...
        ['SimulationParameters.EvaluationCapture.OutputPath is required ' ...
         'when capture is enabled.']);
end

validBubbleCount = numel(MB.radii);
requestedBubbleCount = validBubbleCount;
if isfield(config, 'RequestedBubbleCount')
    requestedBubbleCount = config.RequestedBubbleCount;
end
if ~isscalar(requestedBubbleCount) || requestedBubbleCount < 1 || ...
        requestedBubbleCount ~= floor(requestedBubbleCount)
    error('main_RF:InvalidRequestedBubbleCount', ...
        'EvaluationCapture.RequestedBubbleCount must be a positive integer.');
end
seededBubbleCount = validBubbleCount;
if isfield(config, 'SeededBubbleCount')
    seededBubbleCount = config.SeededBubbleCount;
end
if validBubbleCount < requestedBubbleCount
    error('main_RF:InsufficientCapturedBubbles', ...
        ['Evaluation capture seeded %d bubble(s), found %d valid in-grid ' ...
         'bubble(s), and requires %d.'], ...
        seededBubbleCount, validBubbleCount, requestedBubbleCount);
end
selectedBubbleIndices = 1:requestedBubbleCount;
sensed_p = sensed_p(selectedBubbleIndices, :);
radii = MB.radii(selectedBubbleIndices);

capture.sensed_p = sensed_p;
capture.radii = radii;
capture.t_kwave = (0:(size(sensed_p, 2) - 1)) * kgrid.dt;
capture.kgrid.dt = kgrid.dt;
capture.kgrid.k_max = get_bubble_filter_kmax(...
    kgrid, Medium, SimulationParameters.HybridSimulation);
capture.Medium = Medium;
capture.Microbubble = Microbubble;
capture.Transmit = Transmit;
capture.frame = frame;
capture.pulse_sequence_index = pulseSequenceIndex;
capture.sensed_p_dtype = class(sensed_p);
capture.hybrid_simulation = logical(SimulationParameters.HybridSimulation);
capture.solver = SimulationParameters.Solver;
capture.device_number = NaN;
if isfield(SimulationParameters, 'DeviceNumber')
    capture.device_number = SimulationParameters.DeviceNumber;
end
capture.seeded_bubble_count = seededBubbleCount;
capture.valid_bubble_count = validBubbleCount;
capture.selected_bubble_count = requestedBubbleCount;
capture.selected_bubble_indices = selectedBubbleIndices;
capture.created_at = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

outputPath = char(config.OutputPath);
outputFolder = fileparts(outputPath);
if ~isempty(outputFolder) && ~exist(outputFolder, 'dir')
    mkdir(outputFolder)
end
save(outputPath, 'capture', '-v7.3')
fprintf('Saved evaluation pressure capture to %s.\n', outputPath)

if isfield(config, 'StopAfterCapture')
    if ~isscalar(config.StopAfterCapture)
        error('main_RF:InvalidStopAfterCapture', ...
            'EvaluationCapture.StopAfterCapture must be scalar.');
    end
    stopAfterCapture = logical(config.StopAfterCapture);
end

end

function context = bubble_solver_context(exception, frame, pulseSequenceIndex)
% Attach frame and pulse context to a failure raised by the bubble solver.

if ~startsWith(exception.identifier, 'compute_bubble_mass_source:') && ...
        ~startsWith(exception.identifier, 'calcBubbleResponse_GPU:')
    context = exception;
    return
end
context = MException('main_RF:BubbleSolverFailed', ...
    ['Bubble solver failed at frame %d, pulse sequence %d. No partial ' ...
     'frame was saved; restart from frame %d.'], ...
    frame, pulseSequenceIndex, frame);
context = addCause(context, exception);
end

function batch_size = get_transmit_batch_size(SimulationParameters, Acquisition)
num_frames = Acquisition.EndFrame - Acquisition.StartFrame + 1;
if num_frames < 1
    error('main_RF:InvalidFrameRange', ...
        'Acquisition.EndFrame must be greater than or equal to StartFrame.')
end

if isfield(SimulationParameters, 'TransmitBatchSize') && ...
        ~isempty(SimulationParameters.TransmitBatchSize)
    batch_size = SimulationParameters.TransmitBatchSize;
else
    batch_size = 50;
end

if isinf(batch_size) || batch_size == 0
    batch_size = num_frames;
end

if ~isnumeric(batch_size) || ~isscalar(batch_size) || ...
        batch_size < 1 || floor(batch_size) ~= batch_size
    error('main_RF:InvalidTransmitBatchSize', ...
        'SimulationParameters.TransmitBatchSize must be a positive integer, Inf, or 0.')
end

batch_size = min(batch_size, num_frames);
end


function frame_batches = make_frame_batches(start_frame, end_frame, batch_size)
num_batches = ceil((end_frame - start_frame + 1) / batch_size);
frame_batches = zeros(num_batches, 2);
for batch_idx = 1:num_batches
    batch_start = start_frame + (batch_idx - 1) * batch_size;
    batch_end = min(end_frame, batch_start + batch_size - 1);
    frame_batches(batch_idx, :) = [batch_start, batch_end];
end
end


function sync_if_on_device(data)
% Timing a stage whose result is still queued on the GPU measures how long it
% took to submit the work, not to do it. Only pay for the barrier when the
% data is actually on the device -- the k-Wave binaries hand back host arrays,
% so on those paths this costs nothing.

if isa(data, 'gpuArray')
    wait(gpuDevice);
end

end


function sensor_subset = extract_sensor_subset(...
    sensor_data, source_mask_idx, target_mask_idx, n_time_points)
[~, sensor_data_idx, ~] = intersect(source_mask_idx, target_mask_idx);
sensor_subset.p = sensor_data.p(sensor_data_idx, 1:n_time_points);
end


function assert_tiling_metadata_matches(FlowSimulationParameters, medium_metadata)
gt_tiling = struct();
if isfield(FlowSimulationParameters, 'Tiling')
    gt_tiling = FlowSimulationParameters.Tiling;
end
if ~isfield(gt_tiling, 'Enabled')
    gt_tiling.Enabled = false;
end
if ~isfield(medium_metadata, 'Tiling')
    medium_tiling.Enabled = false;
else
    medium_tiling = medium_metadata.Tiling;
end
if ~isfield(medium_tiling, 'Enabled')
    medium_tiling.Enabled = false;
end
if gt_tiling.Enabled ~= medium_tiling.Enabled
    error('main_RF:TilingMetadataMismatch', ...
        'Ground-truth tiling metadata does not match medium tiling metadata.')
end
if gt_tiling.Enabled
    if ~isfield(gt_tiling, 'Transforms') || ...
            ~isfield(medium_tiling, 'Transforms') || ...
            numel(gt_tiling.Transforms) ~= numel(medium_tiling.Transforms)
        error('main_RF:TilingMetadataMismatch', ...
            'Tiled GT requires matching tiled medium transform metadata.')
    end
end
end
