function generate_streamlines(Geometry, Microbubble, Acquisition, ...
    PATHS, savefolder, showStreamlines)
% Track microbubbles flowing through the flow vector field given by the vtu
% file in the specified geometry folder. When a bubble reaches an outlet of
% the vessel, a new bubble is generated at the inlet to keep the bubble
% count constant. For each frame, the positions, velocities, stream
% numbers, and radii of the bubbles are stored. The stream number indicates
% how often a bubble has been refreshed (1 corresponding to the first
% streamline, no refreshing).
%
% Nathan Blanken, University of Twente, 2023
% Guillaume Lajoinie, University of Twente, 2023

%==========================================================================
% GET USER PARAMETERS
%==========================================================================

% Folder containing the geometry data:
geometryFolder = [PATHS.GeometriesPath filesep Geometry.Folder];

frameRate  = Acquisition.FrameRate; % [Hz]
NFrames    = Acquisition.NumberOfFrames;
NPulses    = Acquisition.NumberOfPulses;
timeBetweenPulses = Acquisition.TimeBetweenPulses;

% Number of bubbles at each moment in the vessel:
NBubbles   = Microbubble.Number;

% Microbubble size distribution P(R):
P = Microbubble.Distribution.Probabilities;
R = Microbubble.Distribution.Radii;

% Use parallel computing for the microbubble tracking:
if isfield(Acquisition,'ParallelTracking')
    useparfor = Acquisition.ParallelTracking;
else
    useparfor = false;
end

%==========================================================================
% READ VTU DATA AND INLET DATA
%==========================================================================

% MATLAB file with VTU data of the flow simulation:
filename = [geometryFolder filesep 'vtu.mat'];
GeometryPropertiesFilename = ...
    [geometryFolder filesep 'GeometryProperties.mat'];

load(GeometryPropertiesFilename,'vtuProperties')
[vtuStruct, Grid] = load_vessel_data(filename, vtuProperties);

% Load the inlet points:
inlet = load([geometryFolder filesep 'inlet.mat'],'inlet');
inlet = inlet.inlet;

%--------------------------------------------------------------------------
% ODE solver options
%--------------------------------------------------------------------------

load(GeometryPropertiesFilename,'options');
options = odeset(options,'Events',@(t,y)exitVesselFcn(t,y,Grid));

% Velocity scaling factor for MB trajectory integration and effective labels.
% The default is the CFD field as the CFD solved it. This was hardcoded at 5
% until 2026-08-27, so every dataset carried a 5x flow that no settings file
% recorded: the value reached the ODE without passing through Acquisition,
% and a reader holding the config could not have known. A run that wants
% faster flow now has to ask for it, where write_run_manifest and
% check_ground_truth_data can both see the request.
%
% The scale does not touch single-bubble physics -- velocity never reaches
% the bubble solver, only the trajectory integration and the labels. What it
% does set is inter-frame displacement, and that is what makes a dataset
% trackable or not. Measured on run_20260827_082616 at scale 5 and 500 Hz:
% median displacement 218 um against a median nearest-neighbour spacing of
% 131 um, a ratio of 1.66. Below 1 is the trackable regime.
if isfield(Acquisition, 'VelocityScale') && ~isempty(Acquisition.VelocityScale)
    VELOCITY_SCALE = Acquisition.VelocityScale;
    velocityScaleSource = 'Acquisition.VelocityScale';
else
    VELOCITY_SCALE = 1;
    velocityScaleSource = 'default, field absent from settings';
end
validate_velocity_scale(VELOCITY_SCALE, velocityScaleSource);
fprintf('MB velocity scale: %g x CFD field (%s)\n', ...
    VELOCITY_SCALE, velocityScaleSource);

% Bubble start positions and radii were drawn with no rng call, exactly as the
% tissue speckle was, so the ground truth of a run could not be reproduced
% either. Seeding only the medium would have left the pairing of a frame's
% bubbles to its RF unrepeatable, which is the half that carries the labels.
%
% 'shuffle' stays the default so nothing changes silently; the integer the
% stream started from is recorded either way.
requestedSeed = 'shuffle';
if isfield(Acquisition, 'RandomSeed')
    requestedSeed = Acquisition.RandomSeed;
end
if isempty(which('resolve_random_seed'))
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'acoustic-module'));
end
[RANDOM_SEED, randomSeedSource] = ...
    resolve_random_seed(requestedSeed, 'Acquisition.RandomSeed');
fprintf('Ground truth seed: %d (%s)\n', RANDOM_SEED, randomSeedSource);

% Where a bubble that has left the vessel is put back. Upstream reseeded at
% the inlet, which forced every replacement bubble to traverse the tree from
% the feeding vessel down. The tiling rewrite routed the reseed through
% build_tile_problem, which draws from the vessel bulk, and the inlet has
% been dead code since: loaded, passed into track_bubble, never read.
%
% Measured on run_20260827_082616, the two are not close. Reseed positions
% span 11.35 x 5.28 x 16.06 mm against the frame-1 bulk seeds' 11.27 x 5.57
% x 16.31 mm, with centroids 0.39 mm apart -- one cloud, not a cluster at a
% vessel end.
%
% It compounds with velocity-weighted seeding: a bubble starts in the fast
% half of the field and, on exit, is reborn there rather than upstream, so
% the slow vessels are never traversed. Only 1.9% of visited samples in that
% run fall below the seeding cut.
if isfield(Acquisition, 'ReseedFrom') && ~isempty(Acquisition.ReseedFrom)
    RESEED_FROM = lower(char(Acquisition.ReseedFrom));
    reseedFromSource = 'Acquisition.ReseedFrom';
else
    RESEED_FROM = 'inlet';
    reseedFromSource = 'default, field absent from settings';
end
validate_reseed_from(RESEED_FROM, reseedFromSource);
fprintf('MB reseeding: %s (%s)\n', RESEED_FROM, reseedFromSource);

% Random vessel-volume seeding can otherwise pick stagnant/near-stagnant
% cells, which creates MBs that appear fixed across many frames. Weighting
% start-position sampling toward the faster half of the CFD velocity field
% fixes that, at the cost of a dataset that barely visits the slow vessels:
% measured on run_20260827_082616, only 1.9% of visited samples fall below the
% cut, against a median visited speed of 4.6x it. Those are the vessels ULM
% exists to resolve, so the trade is the caller's to make.
%
% This was three literals with no Acquisition hook at all -- not even a read
% path -- until 2026-08-27. It defaults to off, which is upstream's uniform
% sampling over the vessel volume.
SeedCfg.Enabled = false;
SeedCfg.MinSpeedPercentile = 50;
SeedCfg.WeightPower = 1;
if isfield(Acquisition, 'Seeding') && ~isempty(Acquisition.Seeding)
    SeedCfg = merge_struct(SeedCfg, Acquisition.Seeding);
end
[vtuStruct, SeedStats] = apply_velocity_weighted_seeding(vtuStruct, SeedCfg);
if SeedCfg.Enabled
    fprintf(['Velocity-weighted MB seeding: keeping %.1f%% of vessel cells ' ...
        '(min speed %.4g m/s, median %.4g m/s, max %.4g m/s)\n'], ...
        100 * SeedStats.KeptFraction, SeedStats.MinSpeed, ...
        SeedStats.MedianSpeed, SeedStats.MaxSpeed);
end

% Domain-aware seed cropping (general, config-adaptive). When enabled, restrict
% seeding to vessel cells inside the simulated domain box (Geometry.Domain) so
% microbubbles start where they are actually imaged, without hand-tuning tiling
% offset ranges. Pair with zero tiling offsets (the crop is the confinement).
if isfield(Acquisition, 'ConfineSeedsToDomain') && Acquisition.ConfineSeedsToDomain
    [vtuStruct, CropStats] = crop_vessel_to_domain(vtuStruct, Geometry, 0.15);
    fprintf(['Domain-crop seeding: %.1f%% of vessel cells inside domain box ' ...
        '(%d seedable cells)\n'], 100*CropStats.InBoxFraction, ...
        CropStats.SeedableCells);
end

% --- Vessel tiling: replicate the canonical vessel across the imaging FOV
% with a random per-streamline offset (and optional rotation about the
% elevation axis) so MBs cover the whole image plane and flow in different
% directions.
%
% Defaults to off -- upstream has one vessel. It defaulted to ON until
% 2026-08-27 and was off in production only because the driver notebook wrote
% Acquisition.Tiling = struct('Enabled', false) over it. Four runs did use it:
% run_20260702 through run_20260716, at 200 tiles.
TileCfg.Enabled              = false;
TileCfg.RandomizeRotation    = true;          % rotate flow direction in image plane
TileCfg.DepthRange           = [-0.025, 0.002];  % m, image-X (depth) offset range
TileCfg.WidthRange           = [-0.015, 0.015];  % m, image-Y (lateral) offset range
TileCfg.ElevRange            = [-0.0005, 0.0005];% m, image-Z (elevation) offset range
TileCfg.Rotation             = Geometry.Rotation;
TileCfg.BB_center            = reshape(Geometry.BoundingBox.Center, 3, 1);
TileCfg.TransformFrame = 'vessel_to_image_consistent';
TileCfg.RandomSeed           = 0;
TileCfg.NumTiles             = max(1, NBubbles);
if isfield(Acquisition, 'Tiling') && ~isempty(Acquisition.Tiling)
    % v9b, v9c and v9d set offset ranges and no Enabled field, so they ran
    % tiled purely on the old default of true. Flipping that default would
    % have made all three single-vessel runs with nothing said, so an
    % ambiguous struct is refused rather than resolved.
    if ~isfield(Acquisition.Tiling, 'Enabled')
        error('generate_streamlines:TilingEnabledMissing', ...
            ['Acquisition.Tiling was given without an Enabled field. ' ...
             'Tiling defaulted to on until 2026-08-27; state ' ...
             'Acquisition.Tiling.Enabled explicitly.']);
    end
    TileCfg = merge_struct(TileCfg, Acquisition.Tiling);
end
TileCfg.Transforms = build_tile_transforms(TileCfg);

% Canonical (un-tiled) ODE function. Per-streamline tiled odefun is built
% inside track_bubble.
odefun = @(t,y) VELOCITY_SCALE * transpose(...
    get_velocity(transpose(y), Grid, vtuStruct.velocities));


%==========================================================================
% COMPUTE STREAMLINES
%==========================================================================

% Matrices for holding the microbubble positions, velocities, streamline
% counts, and radii:
streamlines   = zeros(NPulses*NFrames, NBubbles,3);
velocities    = zeros(NPulses*NFrames, NBubbles,3);
rawVelocities = zeros(NPulses*NFrames, NBubbles,3);
streamNumbers = zeros(NPulses*NFrames, NBubbles);
tileIDs       = zeros(NPulses*NFrames, NBubbles);
radii         = zeros(NPulses*NFrames, NBubbles);
bubbleIndexes = zeros(NPulses*NFrames, NBubbles);
trackIDs      = zeros(NPulses*NFrames, NBubbles);

t1 = tic;
if showStreamlines; h = figure(); end

if useparfor
    
    %----------------------------------------------------------------------
    % PARALLEL COMPUTING OF STREAMLINES
    %----------------------------------------------------------------------
    
    % Cells for storing the output of the parallel operations:
    streamlines_cell   = cell(1, NBubbles);
    velocities_cell    = cell(1, NBubbles);
    rawVelocities_cell = cell(1, NBubbles);
    streamNumbers_cell = cell(1, NBubbles);
    tileIDs_cell       = cell(1, NBubbles);
    radii_cell         = cell(1, NBubbles);
    bubbleIndexes_cell = cell(1, NBubbles);
    trackIDs_cell      = cell(1, NBubbles);

    parfor n = 1:NBubbles

        disp(['Tracking microbubble ' num2str(n)...
            ' of ' num2str(NBubbles) '.']);

        % Track the bubble:
        [...
            streamlines_cell{   n}, ...
            velocities_cell{    n}, ...
            rawVelocities_cell{ n}, ...
            streamNumbers_cell{ n}, ...
            tileIDs_cell{       n}, ...
            radii_cell{         n}, ...
            bubbleIndexes_cell{ n}, ...
            trackIDs_cell{      n}  ...
            ] = ...
            track_bubble(Microbubble, Acquisition, Grid, ...
            vtuStruct, inlet, odefun, options, showStreamlines, ...
            VELOCITY_SCALE, RESEED_FROM, RANDOM_SEED, TileCfg, n, n, ...
            NBubbles);
    end

    % Assign the streamline values in the cells to the matrices:
    for n = 1:NBubbles
        streamlines(  :, n,:) = streamlines_cell{   n};
        velocities(   :, n,:) = velocities_cell{    n};
        rawVelocities(:, n,:) = rawVelocities_cell{ n};
        streamNumbers(:, n)   = streamNumbers_cell{ n};
        tileIDs(      :, n)   = tileIDs_cell{       n};
        radii(        :, n)   = radii_cell{         n};
        bubbleIndexes(:, n)   = bubbleIndexes_cell{ n};
        trackIDs(     :, n)   = trackIDs_cell{      n};
    end
    
else
    
    %----------------------------------------------------------------------
    % SERIAL COMPUTING OF STREAMLINES
    %----------------------------------------------------------------------
    
    for n = 1:NBubbles

        disp(['Tracking microbubble ' num2str(n)...
            ' of ' num2str(NBubbles) '.']);

        % Track the bubble:
        [...
            streamlines(   :, n, :), ...
            velocities(    :, n, :), ...
            rawVelocities( :, n, :), ...
            streamNumbers( :, n), ...
            tileIDs(       :, n), ...
            radii(         :, n), ...
            bubbleIndexes( :, n), ...
            trackIDs(      :, n) ...
            ] = ...
            track_bubble(Microbubble, Acquisition, Grid, ...
            vtuStruct, inlet, odefun, options, showStreamlines, ...
            VELOCITY_SCALE, RESEED_FROM, RANDOM_SEED, TileCfg, n, n, ...
            NBubbles);

    end
    
end

toc(t1)
if showStreamlines; close(h); end

%==========================================================================
% SAVE DATA
%==========================================================================

disp('Saving data ...')

streamlines   = reshape(streamlines,   NPulses, NFrames, NBubbles, 3);
velocities    = reshape(velocities,    NPulses, NFrames, NBubbles, 3);
rawVelocities = reshape(rawVelocities, NPulses, NFrames, NBubbles, 3);
streamNumbers = reshape(streamNumbers, NPulses, NFrames, NBubbles);
tileIDs       = reshape(tileIDs,       NPulses, NFrames, NBubbles);
radii         = reshape(radii,         NPulses, NFrames, NBubbles);
bubbleIndexes = reshape(bubbleIndexes, NPulses, NFrames, NBubbles);
trackIDs      = reshape(trackIDs,      NPulses, NFrames, NBubbles);

if ~exist([PATHS.GroundTruthPath filesep savefolder],'dir')
    mkdir([PATHS.GroundTruthPath filesep savefolder]);
end

% Save the streamline generation parameters:
FlowSimulationParameters.TimeBtwPulse   = timeBetweenPulses;
FlowSimulationParameters.FrameRate      = frameRate;
FlowSimulationParameters.NBPulses       = NPulses;
FlowSimulationParameters.NMicrobubbles  = NBubbles;
FlowSimulationParameters.NumberOfFrames = NFrames;

FlowSimulationParameters.Microbubble.Distribution.Probabilities = P;
FlowSimulationParameters.Microbubble.Distribution.Radii         = R;
FlowSimulationParameters.VelocityScale = VELOCITY_SCALE;
FlowSimulationParameters.Velocity.Scale = VELOCITY_SCALE;
FlowSimulationParameters.Velocity.RawUnits = 'm/s from CFD VTU field before scaling';
FlowSimulationParameters.Velocity.EffectiveUnits = ...
    'm/s after Velocity.Scale, matching integrated point displacement';
FlowSimulationParameters.Velocity.LabelFieldDefinition = ...
    'Frame.PulseN.Velocity is effective scaled velocity; Frame.PulseN.RawVelocity is unscaled CFD velocity in the same coordinate frame.';
FlowSimulationParameters.Identity.TrackIDFormula = ...
    'bubbleIndex + NBubbles*(streamCount - 1)';
FlowSimulationParameters.Identity.Definition = ...
    ['Frame.PulseN.BubbleIndex is the tracking slot, constant across ' ...
     'frames. A slot is reseeded on a new streamline whenever its bubble ' ...
     'leaves the vessel, and StreamNumber counts those reseeds within the ' ...
     'slot, so (BubbleIndex, StreamNumber) identifies one continuous ' ...
     'track. TrackID is that pair folded into one number and is the field ' ...
     'to group by. Neither StreamNumber nor TileID is unique on its own: ' ...
     'every slot starts at StreamNumber 1, and TileID is shared whenever ' ...
     'tiling is disabled.'];
FlowSimulationParameters.Tiling = TileCfg;
FlowSimulationParameters.Tiling.Transforms = TileCfg.Transforms;
FlowSimulationParameters.Seeding = SeedCfg;
FlowSimulationParameters.Seeding.ReseedFrom = RESEED_FROM;
FlowSimulationParameters.RandomSeed = RANDOM_SEED;
FlowSimulationParameters.RandomSeedSource = randomSeedSource;
FlowSimulationParameters.Seeding.Stats = SeedStats;

save([PATHS.GroundTruthPath, filesep, savefolder, ...
    filesep,'FlowSimulationParameters.mat'],'FlowSimulationParameters');

% Save the ground truth frames:
for m = 1:NFrames
    for n = 1:NPulses

        pulse = ['Pulse' num2str(n)];

        Frame.(pulse).Points       = reshape(streamlines(   n,m,:,:), NBubbles, 3);
        Frame.(pulse).Velocity     = reshape(velocities(    n,m,:,:), NBubbles, 3);
        Frame.(pulse).RawVelocity  = reshape(rawVelocities( n,m,:,:), NBubbles, 3);
        Frame.(pulse).Radius       = reshape(radii(         n,m,:,:), NBubbles, 1);
        Frame.(pulse).StreamNumber = reshape(streamNumbers( n,m,:,:), NBubbles, 1);
        Frame.(pulse).TileID       = reshape(tileIDs(       n,m,:,:), NBubbles, 1);
        Frame.(pulse).BubbleIndex  = reshape(bubbleIndexes( n,m,:,:), NBubbles, 1);
        Frame.(pulse).TrackID      = reshape(trackIDs(      n,m,:,:), NBubbles, 1);

    end
    
    NumOfFramesPadding=num2str(length(num2str(NFrames)));
    save([PATHS.GroundTruthPath,filesep,savefolder,filesep,...
        'Frame_',num2str(m,['%0',NumOfFramesPadding,'i']),'.mat'],'Frame');
end

end



function [streamlines, velocities, rawVelocities, streamNumbers, tileIDs, ...
    radii, bubbleIndexes, trackIDs] = ...
    track_bubble(Microbubble, Acquisition, Grid, vtuStruct, inlet, ...
    odefun, options, showStreamlines, VELOCITY_SCALE, RESEED_FROM, ...
    RANDOM_SEED, TileCfg, initialTileID, bubbleIndex, NBubbles)

% Give the slot its own stream, derived from the run's seed and the slot
% index. Seeding once before the loop would not survive Acquisition.
% ParallelTracking: a parfor worker gets its own generator, so a single rng
% call in the caller leaves the workers as unseeded as before. Deriving the
% stream here makes a slot's trajectory the same whichever worker runs it and
% in whatever order.
% mod because rng rejects a seed at or above 2^32 and a shuffled base seed
% can already sit near it.
rng(mod(RANDOM_SEED + bubbleIndex, 2^32), 'twister');

% The first bubble of a slot starts in the vessel bulk, as upstream does.
% Every replacement after it goes wherever the reseed policy says.
if strcmp(RESEED_FROM, 'inlet')
    reseedStruct = inlet;
else
    reseedStruct = vtuStruct;
end

%--------------------------------------------------------------------------
% GET USER PARAMETERS
%--------------------------------------------------------------------------

frameRate  = Acquisition.FrameRate; % [Hz]
NFrames    = Acquisition.NumberOfFrames;
NPulses    = Acquisition.NumberOfPulses;
timeBetweenPulses = Acquisition.TimeBetweenPulses;

% Time arrays with acquisition times and sequence times:
acquisitionTimes = (0:(NFrames - 1))/frameRate;
sequenceTimes    = (0:(NPulses - 1))*timeBetweenPulses;

% numberOfFrames-by-numberOfPulses time array:
acquisitionTimes = acquisitionTimes + transpose(sequenceTimes);

% Reshape into a row vector:
acquisitionTimes = reshape(acquisitionTimes,1,NPulses*NFrames);

% Microbubble size distribution P(R):
P = Microbubble.Distribution.Probabilities;
R = Microbubble.Distribution.Radii;

streamlines   = zeros(NPulses*NFrames,1,3);
velocities    = zeros(NPulses*NFrames,1,3);
rawVelocities = zeros(NPulses*NFrames,1,3);
streamNumbers = zeros(NPulses*NFrames,1,1);
tileIDs       = zeros(NPulses*NFrames,1,1);
radii         = zeros(NPulses*NFrames,1,1);
bubbleIndexes = zeros(NPulses*NFrames,1,1);
trackIDs      = zeros(NPulses*NFrames,1,1);

% Sample the first streamline tile (per-streamline transform) and produce
% the corresponding tiled odefun, event handler, and start position.
tileID = next_tile_id(TileCfg, initialTileID);
[tileRot, tileOffset] = sample_tile(TileCfg, tileID);
[odefun_eff, options_eff, startPosition] = ...
    build_tile_problem(TileCfg, tileRot, tileOffset, ...
        Grid, vtuStruct, vtuStruct, options, VELOCITY_SCALE, odefun);

tspan = acquisitionTimes;

streamCount = 1; % Streamline count
t = -Inf;

while max(t)<max(acquisitionTimes)

    %------------------------------------------------------------------
    % COMPUTE STREAMLINE
    %------------------------------------------------------------------
    if length(tspan)<2
        t = tspan; positions = startPosition;
    else
        [t,positions] = ode23(odefun_eff,tspan,startPosition(:),options_eff);
    end

    %------------------------------------------------------------------
    % PLOT STREAMLINE
    %------------------------------------------------------------------
    if showStreamlines
        plot3(positions(:,1),positions(:,2),positions(:,3));
        xlabel('X (m)')
        ylabel('Y (m)')
        zlabel('Z (m)')
        hold on
        drawnow
    end

    %------------------------------------------------------------------
    % STORE STREAMLINE
    %------------------------------------------------------------------
    % Find the mutual times in both time arrays:
    [~,I,I_acquisition] = intersect(t,acquisitionTimes);

    streamlines(I_acquisition, 1,:) = positions(I,:);
    streamNumbers(I_acquisition, 1) = streamCount;
    tileIDs(I_acquisition, 1) = tileID;

    % Identity. Slots run under parfor, so the track id has to be a formula
    % over values this worker already holds rather than a shared counter.
    % bubbleIndex spans 1..NBubbles and streamCount starts at 1, so the
    % products never collide.
    bubbleIndexes(I_acquisition, 1) = bubbleIndex;
    trackIDs(I_acquisition, 1) = bubbleIndex + NBubbles*(streamCount - 1);

    % Get the velocities at the microbubble positions. Map the tiled
    % positions back to canonical vessel coords for the lookup, then rotate
    % the looked-up velocity into the tile.
    if TileCfg.Enabled
        canonicalPos = transpose(tileRot' * (positions(I,:)' - ...
            TileCfg.BB_center - tileOffset) + TileCfg.BB_center);
        v_canonical = get_velocity(canonicalPos, Grid, vtuStruct.velocities);
        rawVelocities(I_acquisition, 1, :) = transpose(tileRot * v_canonical');
        velocities(I_acquisition, 1, :) = ...
            VELOCITY_SCALE * transpose(tileRot * v_canonical');
    else
        v_raw = get_velocity(positions(I,:), Grid, vtuStruct.velocities);
        rawVelocities(I_acquisition, 1, :) = v_raw;
        velocities(I_acquisition, 1, :) = VELOCITY_SCALE * v_raw;
    end

    % Draw a radius from the size distribution:
    radii(I_acquisition, 1) = draw_random_radii(P,R,1);

    %------------------------------------------------------------------
    % GET A NEW BUBBLE (fresh tile + start position per streamline)
    %------------------------------------------------------------------
    tileID = next_tile_id(TileCfg, initialTileID + streamCount);
    [tileRot, tileOffset] = sample_tile(TileCfg, tileID);
    [odefun_eff, options_eff, startPosition] = ...
        build_tile_problem(TileCfg, tileRot, tileOffset, ...
            Grid, vtuStruct, reseedStruct, options, VELOCITY_SCALE, odefun);

    % Update time array (remaining time):
    tspan = acquisitionTimes(find(acquisitionTimes>t(end),1):end);

    streamCount = streamCount + 1;

end

end


%==========================================================================
% TILE HELPERS
%==========================================================================

function base = merge_struct(base, override)
% Overlay the fields an Acquisition struct provides onto the defaults. Used
% for both the tiling and the seeding config, so the names stay generic.
fields = fieldnames(override);
for i = 1:numel(fields)
    base.(fields{i}) = override.(fields{i});
end
end


function transforms = build_tile_transforms(TileCfg)
numTiles = max(1, TileCfg.NumTiles);
transforms = repmat(struct( ...
    'TileID', 1, ...
    'Rotation', eye(3), ...
    'Offset', zeros(3, 1), ...
    'TransformFrame', TileCfg.TransformFrame), numTiles, 1);

if ~TileCfg.Enabled
    transforms(1).TileID = 1;
    transforms(1).Rotation = eye(3);
    transforms(1).Offset = zeros(3, 1);
    transforms = transforms(1);
    return
end

rng_state = rng;
rng(TileCfg.RandomSeed);
for tileID = 1:numTiles
    [tileRot, tileOffset] = sample_tile_random(TileCfg);
    transforms(tileID).TileID = tileID;
    transforms(tileID).Rotation = tileRot;
    transforms(tileID).Offset = tileOffset;
    transforms(tileID).TransformFrame = TileCfg.TransformFrame;
end
rng(rng_state);
end


function tileID = next_tile_id(TileCfg, idx)
tileID = mod(idx - 1, numel(TileCfg.Transforms)) + 1;
end


function [tileRot, tileOffset] = sample_tile(TileCfg, tileID)
% Sample a per-streamline tile transform. Returns the vessel-space rotation
% matrix and translation column vector that map canonical vessel positions
% into the tiled position. When tiling is disabled, returns identity / zero.

if isfield(TileCfg, 'Transforms') && ~isempty(TileCfg.Transforms)
    transform = TileCfg.Transforms(tileID);
    tileRot = transform.Rotation;
    tileOffset = transform.Offset;
    return
end

if ~TileCfg.Enabled
    tileRot = eye(3);
    tileOffset = zeros(3, 1);
    return
end
[tileRot, tileOffset] = sample_tile_random(TileCfg);
end


function [tileRot, tileOffset] = sample_tile_random(TileCfg)
if ~TileCfg.Enabled
    tileRot = eye(3);
    tileOffset = zeros(3, 1);
    return
end

% Random offset in image space: [depth; width; elevation]
T_img = [
    rand_in(TileCfg.DepthRange);
    rand_in(TileCfg.WidthRange);
    rand_in(TileCfg.ElevRange)];

% Random rotation about the elevation axis (image z) so the vessel flow
% direction in the imaging plane is randomised per streamline.
if TileCfg.RandomizeRotation
    theta = 2*pi*rand;
    R_theta_img = [cos(theta), -sin(theta), 0;
                   sin(theta),  cos(theta), 0;
                   0,           0,          1];
else
    R_theta_img = eye(3);
end

% Convert from image-space transform to vessel-space transform. The
% vessel-to-image map is image = R_geom * (vessel - BB_center) + Geom.Center,
% so an image-space rotation about Geom.Center by R_theta_img and translation
% by T_img corresponds (in vessel space) to:
%   vessel' = R_v * (vessel - BB_center) + BB_center + T_v
% with R_v = R_geom' * R_theta_img * R_geom and T_v = R_geom' * T_img.
R_geom     = TileCfg.Rotation;
tileRot    = R_geom' * R_theta_img * R_geom;
tileOffset = R_geom' * T_img;

end


function [odefun_eff, options_eff, startPosition] = ...
    build_tile_problem(TileCfg, tileRot, tileOffset, Grid, vtuStruct, ...
                       seedStruct, options, VELOCITY_SCALE, odefun_canonical)
% Build the tile-aware ODE function, ODE event options, and a fresh start
% position for the next streamline.
%
% vtuStruct is the velocity field the ODE integrates; seedStruct is where the
% start position is drawn from. They are the same for a slot's first bubble
% and differ for a reseed under the 'inlet' policy, which is why the caller
% passes both rather than this deciding.

if ~TileCfg.Enabled
    odefun_eff    = odefun_canonical;
    options_eff   = options;
    startPosition = draw_start_position(1, seedStruct);
    return
end

BB = TileCfg.BB_center;

% Map a tiled vessel-space point back to canonical vessel-space:
to_canonical = @(y) tileRot' * (y - BB - tileOffset) + BB;

% ODE in tiled space: lookup velocity at canonical position, rotate result
% into the tile.
odefun_eff = @(t, y) VELOCITY_SCALE * (tileRot * transpose( ...
    get_velocity(transpose(to_canonical(y)), Grid, vtuStruct.velocities)));

% Event detection still uses the canonical Grid -- transform y first.
options_eff = odeset(options, 'Events', ...
    @(t, y) exitVesselFcn(t, to_canonical(y), Grid));

% Sample a canonical start position then transform it into the tile.
canonical_start = draw_start_position(1, seedStruct);  % 1x3 row
startPosition = transpose( ...
    tileRot * (canonical_start' - BB) + BB + tileOffset);

end


function r = rand_in(range)
r = range(1) + rand * (range(2) - range(1));
end


function [vtuStruct, stats] = apply_velocity_weighted_seeding(vtuStruct, cfg)
% Add vtuStruct.density so draw_start_position samples moving vessel cells.

speed = vecnorm(vtuStruct.velocities, 2, 2);
positive = speed(speed > 0);

stats.MinSpeed = 0;
stats.MedianSpeed = 0;
stats.MaxSpeed = 0;
stats.KeptFraction = 1;

if isempty(positive)
    vtuStruct = rmfield_if_exists(vtuStruct, 'density');
    return
end

sorted_speed = sort(positive);
stats.MedianSpeed = sorted_speed(max(1, ceil(0.50 * numel(sorted_speed))));
stats.MaxSpeed = sorted_speed(end);

if ~cfg.Enabled
    vtuStruct = rmfield_if_exists(vtuStruct, 'density');
    return
end

pct = min(max(cfg.MinSpeedPercentile, 0), 100);
idx = max(1, ceil((pct / 100) * numel(sorted_speed)));
stats.MinSpeed = sorted_speed(idx);

density = speed .^ cfg.WeightPower;
density(speed < stats.MinSpeed) = 0;

if sum(density) <= 0
    density = speed;
end
if sum(density) <= 0
    vtuStruct = rmfield_if_exists(vtuStruct, 'density');
    return
end

vtuStruct.density = density / sum(density);
stats.KeptFraction = mean(vtuStruct.density > 0);
end


function s = rmfield_if_exists(s, name)
if isfield(s, name)
    s = rmfield(s, name);
end
end


function validate_velocity_scale(scale, source)
% A scale that is zero, negative, or non-finite integrates to a silently
% wrong trajectory rather than to an error, and the ground truth would look
% well-formed afterwards. Reject it where the value enters instead.

if ~isnumeric(scale) || ~isscalar(scale) || ~isfinite(scale) || scale <= 0
    error('generate_streamlines:InvalidVelocityScale', ...
        ['Velocity scale must be a positive finite scalar, got %s ' ...
         'from %s.'], mat2str(scale), source);
end

end


function validate_reseed_from(policy, source)
% Reject an unrecognised reseed policy rather than silently falling through
% to the vessel bulk, which is the behaviour this field exists to make
% visible.

allowed = {'inlet', 'bulk'};
if ~ismember(policy, allowed)
    error('generate_streamlines:InvalidReseedFrom', ...
        ['Reseed policy must be one of %s, got ''%s'' from %s.'], ...
        strjoin(allowed, ', '), policy, source);
end

end
