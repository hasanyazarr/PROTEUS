function generate_random_walk_fov(Geometry, Microbubble, Acquisition, ...
    PATHS, savefolder, showStreamlines)
%GENERATE_RANDOM_WALK_FOV Generate full-FOV synthetic MB ground truth.
%
% This generator writes the same Frame_*.mat and
% FlowSimulationParameters.mat files as generate_streamlines, but it does
% not constrain bubbles to the vessel VTU. It is intended for data
% generation runs where MBs should cover the full ultrasound image plane.

%==========================================================================
% USER PARAMETERS
%==========================================================================

frameRate  = Acquisition.FrameRate; % [Hz]
NFrames    = Acquisition.NumberOfFrames;
NPulses    = Acquisition.NumberOfPulses;
timeBetweenPulses = Acquisition.TimeBetweenPulses;
NBubbles   = Microbubble.Number;

P = reshape(Microbubble.Distribution.Probabilities, 1, []);
R = reshape(Microbubble.Distribution.Radii, 1, []);

cfg = random_walk_defaults(Geometry, Acquisition);

if ~isempty(cfg.Seed)
    rng(cfg.Seed, 'twister');
end

if ~exist([PATHS.GroundTruthPath filesep savefolder], 'dir')
    mkdir([PATHS.GroundTruthPath filesep savefolder]);
end

%==========================================================================
% TIME AXIS
%==========================================================================

acquisitionTimes = (0:(NFrames - 1))/frameRate;
sequenceTimes    = (0:(NPulses - 1))*timeBetweenPulses;
acquisitionTimes = acquisitionTimes + transpose(sequenceTimes);
acquisitionTimes = reshape(acquisitionTimes, 1, NPulses*NFrames);
Nsamples = numel(acquisitionTimes);

%==========================================================================
% INITIAL CONDITIONS
%==========================================================================

pos = zeros(NBubbles, 3);
pos(:,1) = rand_range(NBubbles, cfg.AxialRange);
pos(:,2) = rand_range(NBubbles, cfg.LateralRange);
pos(:,3) = rand_range(NBubbles, cfg.ElevationRange);

vel = random_velocity(NBubbles, cfg);
radii0 = draw_visible_radii(P, R, NBubbles, cfg.RadiusRange);

positions_acoustic = zeros(Nsamples, NBubbles, 3);
velocities_acoustic = zeros(Nsamples, NBubbles, 3);
radii = repmat(reshape(radii0, 1, NBubbles), Nsamples, 1);
streamNumbers = ones(Nsamples, NBubbles);

positions_acoustic(1,:,:) = pos;
velocities_acoustic(1,:,:) = vel;

%==========================================================================
% RANDOM WALK WITH REFLECTIVE BOUNDARIES
%==========================================================================

for k = 2:Nsamples
    dt = acquisitionTimes(k) - acquisitionTimes(k-1);
    if dt <= 0
        dt = 1/frameRate;
    end

    if cfg.DirectionJitter > 0
        vel = (1 - cfg.DirectionJitter) * vel + ...
            cfg.DirectionJitter * random_velocity(NBubbles, cfg);
        vel = clamp_velocity(vel, cfg);
    end

    pos = pos + vel * dt;
    [pos, vel] = reflect_bounds(pos, vel, cfg);

    positions_acoustic(k,:,:) = pos;
    velocities_acoustic(k,:,:) = vel;
end

if showStreamlines
    figure;
    hold on;
    for n = 1:NBubbles
        p = squeeze(positions_acoustic(:,n,:));
        plot3(p(:,1), p(:,2), p(:,3));
    end
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Random-walk full-FOV microbubble tracks');
    axis equal;
    drawnow;
end

% Convert from acoustic coordinates back to Geometry coordinates because
% load_microbubbles applies the Geometry transform before RF simulation.
positions_raw = acoustic_to_geometry_points(positions_acoustic, Geometry);
velocities_raw = acoustic_to_geometry_velocity(velocities_acoustic, Geometry);

%==========================================================================
% SAVE DATA
%==========================================================================

positions_raw = reshape(positions_raw, NPulses, NFrames, NBubbles, 3);
velocities_raw = reshape(velocities_raw, NPulses, NFrames, NBubbles, 3);
streamNumbers = reshape(streamNumbers, NPulses, NFrames, NBubbles);
radii = reshape(radii, NPulses, NFrames, NBubbles);

FlowSimulationParameters.TimeBtwPulse   = timeBetweenPulses;
FlowSimulationParameters.FrameRate      = frameRate;
FlowSimulationParameters.NBPulses       = NPulses;
FlowSimulationParameters.NMicrobubbles  = NBubbles;
FlowSimulationParameters.NumberOfFrames = NFrames;

FlowSimulationParameters.Microbubble.Distribution.Probabilities = P;
FlowSimulationParameters.Microbubble.Distribution.Radii         = R;
FlowSimulationParameters.MotionModel = 'random_walk_fov';
FlowSimulationParameters.RandomWalkFOV = cfg;

save([PATHS.GroundTruthPath, filesep, savefolder, ...
    filesep, 'FlowSimulationParameters.mat'], 'FlowSimulationParameters');

for m = 1:NFrames
    Frame = struct();
    for n = 1:NPulses
        pulse = ['Pulse' num2str(n)];

        Frame.(pulse).Points       = reshape(positions_raw(n,m,:,:), NBubbles, 3);
        Frame.(pulse).Velocity     = reshape(velocities_raw(n,m,:,:), NBubbles, 3);
        Frame.(pulse).Radius       = reshape(radii(n,m,:), NBubbles, 1);
        Frame.(pulse).StreamNumber = reshape(streamNumbers(n,m,:), NBubbles, 1);
    end

    NumOfFramesPadding = num2str(length(num2str(NFrames)));
    save([PATHS.GroundTruthPath, filesep, savefolder, filesep, ...
        'Frame_', num2str(m, ['%0', NumOfFramesPadding, 'i']), '.mat'], ...
        'Frame');
end

fprintf('=== Random-walk full-FOV ground truth generated ===\n');
fprintf('  Axial:     [%.1f, %.1f] mm\n', cfg.AxialRange*1e3);
fprintf('  Lateral:   [%.1f, %.1f] mm\n', cfg.LateralRange*1e3);
fprintf('  Elevation: [%.1f, %.1f] mm\n', cfg.ElevationRange*1e3);

end

%==========================================================================
% HELPERS
%==========================================================================

function cfg = random_walk_defaults(Geometry, Acquisition)

D = Geometry.Domain;

domain_margin = 1.0e-3;
x_min = max(D.Xmin + domain_margin, 1.0e-3);
x_max = D.Xmax - domain_margin;
y_min = D.Ymin + domain_margin;
y_max = D.Ymax - domain_margin;

z_half = min((D.Zmax - D.Zmin) / 4, 1.0e-3);
z_center = (D.Zmin + D.Zmax) / 2;

cfg.AxialRange = [x_min x_max];
cfg.LateralRange = [y_min y_max];
cfg.ElevationRange = [z_center - z_half, z_center + z_half];
cfg.SpeedRange = [0.010 0.040];       % [m/s]
cfg.DirectionJitter = 0.20;           % 0 keeps straight tracks, 1 is memoryless
cfg.RadiusRange = [1.2e-6 3.5e-6];    % avoid very dim, very small bubbles
cfg.Seed = [];

if isfield(Acquisition, 'RandomWalkFOV')
    user_cfg = Acquisition.RandomWalkFOV;
    names = fieldnames(user_cfg);
    for k = 1:numel(names)
        cfg.(names{k}) = user_cfg.(names{k});
    end
end

cfg.AxialRange = validate_range(cfg.AxialRange, [D.Xmin D.Xmax], 'AxialRange');
cfg.LateralRange = validate_range(cfg.LateralRange, [D.Ymin D.Ymax], 'LateralRange');
cfg.ElevationRange = validate_range(cfg.ElevationRange, [D.Zmin D.Zmax], 'ElevationRange');
cfg.SpeedRange = validate_range(cfg.SpeedRange, [0 Inf], 'SpeedRange');
cfg.RadiusRange = validate_range(cfg.RadiusRange, [0 Inf], 'RadiusRange');

end

function range = validate_range(range, bounds, name)

range = reshape(range, 1, []);
if numel(range) ~= 2 || any(isnan(range)) || range(1) >= range(2)
    error('Acquisition.RandomWalkFOV.%s must be a valid [min max] range.', name);
end

range(1) = max(range(1), bounds(1));
range(2) = min(range(2), bounds(2));
if range(1) >= range(2)
    error('Acquisition.RandomWalkFOV.%s is outside the simulation domain.', name);
end

end

function x = rand_range(N, range)
x = range(1) + rand(N, 1) * (range(2) - range(1));
end

function vel = random_velocity(N, cfg)

speed = rand_range(N, cfg.SpeedRange);
theta = 2*pi*rand(N, 1);
vel = zeros(N, 3);
vel(:,1) = speed .* cos(theta);
vel(:,2) = speed .* sin(theta);

% Keep elevation motion small so bubbles stay inside the imaging slice.
vel(:,3) = 0.10 * speed .* (2*rand(N, 1) - 1);

end

function vel = clamp_velocity(vel, cfg)

speed_xy = sqrt(vel(:,1).^2 + vel(:,2).^2);
speed_xy(speed_xy == 0) = cfg.SpeedRange(1);
target_speed = min(max(speed_xy, cfg.SpeedRange(1)), cfg.SpeedRange(2));
scale = target_speed ./ speed_xy;
vel(:,1) = vel(:,1) .* scale;
vel(:,2) = vel(:,2) .* scale;
vel(:,3) = min(max(vel(:,3), -0.10*cfg.SpeedRange(2)), 0.10*cfg.SpeedRange(2));

end

function [pos, vel] = reflect_bounds(pos, vel, cfg)

ranges = [cfg.AxialRange; cfg.LateralRange; cfg.ElevationRange];
for dim = 1:3
    lo = ranges(dim, 1);
    hi = ranges(dim, 2);

    below = pos(:,dim) < lo;
    pos(below,dim) = lo + (lo - pos(below,dim));
    vel(below,dim) = abs(vel(below,dim));

    above = pos(:,dim) > hi;
    pos(above,dim) = hi - (pos(above,dim) - hi);
    vel(above,dim) = -abs(vel(above,dim));

    pos(:,dim) = min(max(pos(:,dim), lo), hi);
end

end

function radii = draw_visible_radii(P, R, N, radius_range)

mask = R >= radius_range(1) & R <= radius_range(2) & P > 0;
if ~any(mask)
    warning('No radii inside Acquisition.RandomWalkFOV.RadiusRange; using full distribution.');
    mask = P > 0;
end

P_eff = P(mask);
P_eff = P_eff / sum(P_eff);
R_eff = R(mask);
radii = draw_random_radii(P_eff, R_eff, N);

end

function points_raw = acoustic_to_geometry_points(points_acoustic, Geometry)

sz = size(points_acoustic);
points = reshape(points_acoustic, [], 3);
center = reshape(Geometry.Center, 1, 3);
bbox_center = reshape(Geometry.BoundingBox.Center, 1, 3);

points = (points - center) * Geometry.Rotation + bbox_center;
points_raw = reshape(points, sz);

end

function velocities_raw = acoustic_to_geometry_velocity(velocities_acoustic, Geometry)

sz = size(velocities_acoustic);
velocities = reshape(velocities_acoustic, [], 3);
velocities = velocities * Geometry.Rotation;
velocities_raw = reshape(velocities, sz);

end
