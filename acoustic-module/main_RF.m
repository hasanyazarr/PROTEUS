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

% Timer for frames+MB part (after initial transmit):
disp('=== Starting frames + MB part (timer started) ===');
t_frames_start = tic;
tstart = tic;
execution_times = zeros(1,Acquisition.NumberOfFrames);
saveExecutionTimes = false;

num_frames_to_process = Acquisition.EndFrame - Acquisition.StartFrame + 1;

if SimulationParameters.HybridSimulation
    transmit_batch_size = get_transmit_batch_size(...
        SimulationParameters, Acquisition);
    frame_batches = make_frame_batches(Acquisition.StartFrame, Acquisition.EndFrame, transmit_batch_size);

    sensor_data_transducer_1iter = cell(1,length(sequence));
    kgrid.Nt = floor(run_param.tr(3) / kgrid.dt) + 1;

    for pulse_seq_idx = 1 : length(sequence)
        % Simulation time and memory estimation:
        if pulse_seq_idx == 1 && estimate == true
            beta_coeff_file = ['time-estimation' filesep 'beta_coeff.mat'];
            estim_time_mem(Grid, source_transducer{pulse_seq_idx}, param, ...
                beta_coeff_file);
        end

        disp('Simulating transducer-only transmit wave.')
        t_tx = tic;
        sensor_data_transducer_1iter{pulse_seq_idx} = run_simulation(...
            run_param, kgrid, medium, source_transducer{pulse_seq_idx}, ...
            sensor_transducer);
        fprintf('[TIMING] Transducer transmit wave (pulse %d): %.2f s\n', ...
            pulse_seq_idx, toc(t_tx));
    end

    for batch_idx = 1:size(frame_batches, 1)
        batch_start = frame_batches(batch_idx, 1);
        batch_end = frame_batches(batch_idx, 2);
        fprintf('=== MB transmit batch %d/%d: frames %d-%d ===\n', ...
            batch_idx, size(frame_batches, 1), batch_start, batch_end);

        AcquisitionBatch = Acquisition;
        AcquisitionBatch.StartFrame = batch_start;
        AcquisitionBatch.EndFrame = batch_end;
        [sensor_MB_batch, MB_idx_all, max_mb_batch] = define_sensor_MB_all(...
            Grid, groundtruthfolder, AcquisitionBatch, length(sequence), ...
            Geometry);
        param.max_mb = max(param.max_mb, max_mb_batch);

        if ~isempty(intersect(MB_idx_all, mask_idx_trans))
            warning('Microbubbles on transducer.')
        end

        mask_idx_MB_batch = find(logical(sensor_MB_batch.mask));
        sensor_data_MB_1iter = cell(1,length(sequence));
        kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;

        for pulse_seq_idx = 1 : length(sequence)
            disp('Simulating MB-only transmit wave.')
            t_tx = tic;
            sensor_data_MB_1iter{pulse_seq_idx} = run_simulation(...
                run_param, kgrid, medium, source_transducer{pulse_seq_idx}, ...
                sensor_MB_batch);
            fprintf('[TIMING] MB transmit wave (pulse %d, frames %d-%d): %.2f s\n', ...
                pulse_seq_idx, batch_start, batch_end, toc(t_tx));
        end

        for frame = batch_start : batch_end
            display(['frame ', num2str(frame)])

            RF = cell(1,length(sequence));
            Frame = cell(1,length(sequence));

            for pulse_seq_idx = 1 : length(sequence)

                MB = load_microbubbles(groundtruthfolder, frame, pulse_seq_idx, ...
                    Geometry, Acquisition.NumberOfFrames);

                % define the sensor of the current frame
                [sensor_frame, sensor_weights_frame, MB, run_param.max_dist] = ...
                    define_sensor_MB(Grid, MB);

                mask_idx_frame = find(logical(sensor_frame.mask));
                n_mb_time = floor(run_param.tr(1) / kgrid.dt) + 1;

                sensor_data_MB = extract_sensor_subset(...
                    sensor_data_MB_1iter{pulse_seq_idx}, ...
                    mask_idx_MB_batch, mask_idx_frame, n_mb_time);
                sensor_data_trans = sensor_data_transducer_1iter{pulse_seq_idx};

                % Pressure sensed by the microbubbles
                sensed_p = sensor_weights_frame*double(sensor_data_MB.p);
                sensed_p = cast(full(sensed_p),class(sensor_data_MB.p));

                % Complete the transducer sensor data with microbubble sources:
                sensor_data = hybrid_simulator(...
                    mask_idx_trans,...
                    sensed_p, ...
                    MB, Grid, medium, run_param, ...
                    Medium, Microbubble, Transmit);

                % Update sensor data transducer:
                sensor_data.p = sensor_data_trans.p + sensor_data.p;

                % Compute element RF data recorded by transducer:
                t_rf = tic;
                [RF{pulse_seq_idx}, run_param] = compute_RF_data(...
                    Transducer,sensor_data,sensor_weights,Grid,run_param);
                fprintf('[TIMING] compute_RF_data: %.2f s\n', toc(t_rf));

                Frame{pulse_seq_idx} = MB;

            end

            % Save data
            dt = kgrid.dt;
            % Find out how many zero padding you'll need for file name
            num_padding=num2str(length(num2str(Acquisition.NumberOfFrames)));
            file_name = ['Frame_', num2str(frame,['%0',num_padding,'i']),'.mat'];
            save([savedir filesep file_name], 'RF', 'dt', 'Frame')

            execution_times(frame) = toc(tstart);

        end
    end
else
    [sensor_MB_all, MB_idx_all, max_mb] = define_sensor_MB_all(...
        Grid, groundtruthfolder, Acquisition, length(sequence), Geometry);
    param.max_mb = max_mb;

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
        display(['frame ', num2str(frame)])

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

            sensor_data = full_simulator(...
                source_transducer{pulse_seq_idx}, ...
                sensor_transducer,...
                sensor_frame,sensor_weights_frame,mask_idx_frame,...
                sensed_p,...
                MB, kgrid, Grid, medium, run_param, ...
                Medium, Microbubble, Transmit);

            % Compute element RF data recorded by transducer:
            t_rf = tic;
            [RF{pulse_seq_idx}, run_param] = compute_RF_data(...
                Transducer,sensor_data,sensor_weights,Grid,run_param);
            fprintf('[TIMING] compute_RF_data: %.2f s\n', toc(t_rf));

            Frame{pulse_seq_idx} = MB;

        end

        % Save data
        dt = kgrid.dt;
        % Find out how many zero padding you'll need for file name
        num_padding=num2str(length(num2str(Acquisition.NumberOfFrames)));
        file_name = ['Frame_', num2str(frame,['%0',num_padding,'i']),'.mat'];
        save([savedir filesep file_name], 'RF', 'dt', 'Frame')

        execution_times(frame) = toc(tstart);

    end
end

% Report time for frames + MB part
frames_elapsed = toc(t_frames_start);
disp('=== Frames + MB part complete ===');
fprintf('  Total time:     %.2f s (%.2f min)\n', frames_elapsed, frames_elapsed / 60);
fprintf('  Frames:         %d\n', num_frames_to_process);
fprintf('  Microbubbles:   %d\n', Microbubble.Number);
fprintf('  Per frame:      %.2f s\n', frames_elapsed / num_frames_to_process);

% Save execution times for performance quantification if requested:
if saveExecutionTimes == true
    file_name = 'execution_time_history.mat';
    save([savedir filesep file_name], 'execution_times')
end

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
