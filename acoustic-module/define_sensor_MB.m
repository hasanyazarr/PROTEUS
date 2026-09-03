function [sensor, sensor_weights, MB, max_dist] = ...
    define_sensor_MB(Grid, MB, th)
% =========================================================================
% DEFINE THE SENSOR: MBs and transducer record pressure
% input:    kgrid
%           grid_coordinates
%           folder 
%           frame
%           Geometry
% output:   sensor - mask of sensors
%           MB - struct with linear indexes of MBs in the sensor mask and 
%           indexes of recorded pressure lines (corresponding to MBs) in 
%           sensor_data
%           transducer - update of the transducer struct with same indexes
%           as MBs
% 
%
% th: truncation radius of the band-limited delta function, optional,
%     default 4. See update_sensor.
%
% =========================================================================

if nargin < 3
    th = [];
end

% Logical, not double -- see define_sensor_MB_all. update_sensor assigns
% 1/true into it and every reader takes find() or logical() of it.
sensor.mask = zeros(Grid.Nx, Grid.Ny, Grid.Nz, 'logical');

% Put the microbubbles on the grid:
n_bubbles = size(MB.points, 1);
[MB.points, MB.nodes, MB.idx, idx_exclude] = ...
    voxelize_media_points(MB.points,Grid);

% Exclude microbubbles outside the grid, from every per-bubble field.
%
% This used to name radii and velocities. Everything else load_microbubbles
% reads -- TileID since the tiling work, RawVelocity before it -- kept its
% pre-exclusion length, so from the first dropped bubble on, Frame.TileID(k)
% described a different bubble than Frame.Points(k,:). main_RF saves this
% struct straight into the RF frame, so the mislabelling reached the data:
% 10 of 40 pulses in run_20260827_120645 and 5 of 120 in run_20260831_142721
% carry it. Pruning by row count rather than by name means the next field
% added upstream cannot be forgotten here.
MB = exclude_bubbles(MB, idx_exclude, n_bubbles);

% Compute pair-wise distance matrix:
dx = MB.points(:,1) - transpose(MB.points(:,1));
dy = MB.points(:,2) - transpose(MB.points(:,2));
dz = MB.points(:,3) - transpose(MB.points(:,3));

dist = sqrt(dx.^2 + dy.^2 + dz.^2);

% Maximum distance between pairs of microbubbles:
max_dist = max(dist, [], 'all'); 

% If no microbubbles present, set max_dist to zero:
if isempty(max_dist)
    max_dist = 0;
end

% Put sensors at the microbubbles
mask_only = false;
[sensor, sensor_weights] = update_sensor(sensor, MB.points, MB.idx, ...
    Grid, mask_only, th);

sensor.record={'p'};  

end


function MB = exclude_bubbles(MB, idx_exclude, n_bubbles)
% Drop the excluded rows from every field that has one row per bubble.
%
% points, nodes and idx come back from voxelize_media_points already
% pruned, so they no longer have n_bubbles rows and are skipped by the
% row-count test itself.

if ~any(idx_exclude)
    return
end

fields = fieldnames(MB);
for k = 1:numel(fields)
    value = MB.(fields{k});
    if size(value, 1) == n_bubbles
        value(idx_exclude, :) = [];
        MB.(fields{k}) = value;
    end
end

end
