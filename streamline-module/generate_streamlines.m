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

% Velocity scaling factor (increase to speed up MB flow):
VELOCITY_SCALE = 5;

% --- Vessel tiling: replicate the canonical vessel across the imaging FOV
% with a random per-streamline offset (and optional rotation about the
% elevation axis) so MBs cover the whole image plane and flow in different
% directions. Set ENABLE_TILING=false to restore single-vessel behaviour.
TileCfg.Enabled              = true;
TileCfg.RandomizeRotation    = true;          % rotate flow direction in image plane
TileCfg.DepthRange           = [-0.025, 0.002];  % m, image-X (depth) offset range
TileCfg.WidthRange           = [-0.015, 0.015];  % m, image-Y (lateral) offset range
TileCfg.ElevRange            = [-0.0005, 0.0005];% m, image-Z (elevation) offset range
TileCfg.Rotation             = Geometry.Rotation;
TileCfg.BB_center            = reshape(Geometry.BoundingBox.Center, 3, 1);

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
streamNumbers = zeros(NPulses*NFrames, NBubbles);
radii         = zeros(NPulses*NFrames, NBubbles);

t1 = tic;
if showStreamlines; h = figure(); end

if useparfor
    
    %----------------------------------------------------------------------
    % PARALLEL COMPUTING OF STREAMLINES
    %----------------------------------------------------------------------
    
    % Cells for storing the output of the parallel operations:
    streamlines_cell   = cell(1, NBubbles);
    velocities_cell    = cell(1, NBubbles);
    streamNumbers_cell = cell(1, NBubbles);
    radii_cell         = cell(1, NBubbles);

    parfor n = 1:NBubbles

        disp(['Tracking microbubble ' num2str(n)...
            ' of ' num2str(NBubbles) '.']);

        % Track the bubble:
        [...
            streamlines_cell{   n}, ...
            velocities_cell{    n}, ...
            streamNumbers_cell{ n}, ...
            radii_cell{         n}  ...
            ] = ...
            track_bubble(Microbubble, Acquisition, Grid, ...
            vtuStruct, inlet, odefun, options, showStreamlines, ...
            VELOCITY_SCALE, TileCfg);
    end

    % Assign the streamline values in the cells to the matrices:
    for n = 1:NBubbles
        streamlines(  :, n,:) = streamlines_cell{   n};
        velocities(   :, n,:) = velocities_cell{    n};
        streamNumbers(:, n)   = streamNumbers_cell{ n};
        radii(        :, n)   = radii_cell{         n};
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
            streamNumbers( :, n), ...
            radii(         :, n) ...
            ] = ...
            track_bubble(Microbubble, Acquisition, Grid, ...
            vtuStruct, inlet, odefun, options, showStreamlines, ...
            VELOCITY_SCALE, TileCfg);

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
streamNumbers = reshape(streamNumbers, NPulses, NFrames, NBubbles);
radii         = reshape(radii,         NPulses, NFrames, NBubbles);

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

save([PATHS.GroundTruthPath, filesep, savefolder, ...
    filesep,'FlowSimulationParameters.mat'],'FlowSimulationParameters');

% Save the ground truth frames:
for m = 1:NFrames
    for n = 1:NPulses

        pulse = ['Pulse' num2str(n)];

        Frame.(pulse).Points       = reshape(streamlines(   n,m,:,:), NBubbles, 3);
        Frame.(pulse).Velocity     = reshape(velocities(    n,m,:,:), NBubbles, 3);
        Frame.(pulse).Radius       = reshape(radii(         n,m,:,:), NBubbles, 1);
        Frame.(pulse).StreamNumber = reshape(streamNumbers( n,m,:,:), NBubbles, 1);

    end
    
    NumOfFramesPadding=num2str(length(num2str(NFrames)));
    save([PATHS.GroundTruthPath,filesep,savefolder,filesep,...
        'Frame_',num2str(m,['%0',NumOfFramesPadding,'i']),'.mat'],'Frame');
end

end



function [streamlines, velocities, streamNumbers, radii] = ...
    track_bubble(Microbubble, Acquisition, Grid, vtuStruct, inlet, ...
    odefun, options, showStreamlines, VELOCITY_SCALE, TileCfg)

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
streamNumbers = zeros(NPulses*NFrames,1,1);
radii         = zeros(NPulses*NFrames,1,1);

% Sample the first streamline tile (per-streamline transform) and produce
% the corresponding tiled odefun, event handler, and start position.
[tileRot, tileOffset] = sample_tile(TileCfg);
[odefun_eff, options_eff, startPosition] = ...
    build_tile_problem(TileCfg, tileRot, tileOffset, ...
        Grid, vtuStruct, options, VELOCITY_SCALE, odefun);

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

    % Get the velocities at the microbubble positions. Map the tiled
    % positions back to canonical vessel coords for the lookup, then rotate
    % the looked-up velocity into the tile.
    if TileCfg.Enabled
        canonicalPos = transpose(tileRot' * (positions(I,:)' - ...
            TileCfg.BB_center - tileOffset) + TileCfg.BB_center);
        v_canonical = get_velocity(canonicalPos, Grid, vtuStruct.velocities);
        velocities(I_acquisition, 1, :) = transpose(tileRot * v_canonical');
    else
        velocities(I_acquisition, 1, :) = get_velocity(...
            positions(I,:), Grid, vtuStruct.velocities);
    end

    % Draw a radius from the size distribution:
    radii(I_acquisition, 1) = draw_random_radii(P,R,1);

    %------------------------------------------------------------------
    % GET A NEW BUBBLE (fresh tile + start position per streamline)
    %------------------------------------------------------------------
    [tileRot, tileOffset] = sample_tile(TileCfg);
    [odefun_eff, options_eff, startPosition] = ...
        build_tile_problem(TileCfg, tileRot, tileOffset, ...
            Grid, vtuStruct, options, VELOCITY_SCALE, odefun);

    % Update time array (remaining time):
    tspan = acquisitionTimes(find(acquisitionTimes>t(end),1):end);

    streamCount = streamCount + 1;

end

end


%==========================================================================
% TILE HELPERS
%==========================================================================

function [tileRot, tileOffset] = sample_tile(TileCfg)
% Sample a per-streamline tile transform. Returns the vessel-space rotation
% matrix and translation column vector that map canonical vessel positions
% into the tiled position. When tiling is disabled, returns identity / zero.

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
                       options, VELOCITY_SCALE, odefun_canonical)
% Build the tile-aware ODE function, ODE event options, and a fresh start
% position for the next streamline.

if ~TileCfg.Enabled
    odefun_eff    = odefun_canonical;
    options_eff   = options;
    startPosition = draw_start_position(1, vtuStruct);
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
canonical_start = draw_start_position(1, vtuStruct);  % 1x3 row
startPosition = transpose( ...
    tileRot * (canonical_start' - BB) + BB + tileOffset);

end


function r = rand_in(range)
r = range(1) + rand * (range(2) - range(1));
end