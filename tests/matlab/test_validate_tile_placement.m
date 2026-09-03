repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'streamline-module'));

R_geom = [0 0 -1; -1 0 0; 0 1 0];
Geom = struct( ...
    'Rotation', R_geom, ...
    'Center', [0.0342; 0; 0], ...
    'BoundingBox', struct('Center', [0.008; 0.007; 0.011]), ...
    'Domain', struct('Xmin', 0, 'Xmax', 0.044, ...
                     'Ymin', -0.0226, 'Ymax', 0.0226, ...
                     'Zmin', -0.00358, 'Zmax', 0.00358));

img2vessel = @(img) (R_geom' * (img.' - Geom.Center) + Geom.BoundingBox.Center).';

% A vessel spanning 28-40 mm in depth, +-4 mm laterally, +-2 mm in elevation.
% Its in-plane radius about Geometry.Center is hypot(6.2, 4) = 7.38 mm.
imgPts = [0.028 -0.004 -0.002; 0.040 0.004 0.002; 0.034 0.000 0.000];
vtu = struct('points', img2vessel(imgPts));

% Tile transforms are stored in the vessel frame:
%   R_v = R'*R_theta*R,  T_v = R'*T_img
tile = @(T_img, theta) struct( ...
    'TileID', 1, ...
    'Rotation', R_geom' * [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1] * R_geom, ...
    'Offset', R_geom' * T_img(:), ...
    'TransformFrame', 'vessel_to_image_consistent');

base = struct('Enabled', true, 'RandomizeRotation', true, 'Rotation', R_geom);

%% Tiling that is off is never a placement problem.
off = base; off.Enabled = false; off.Transforms = tile([0.5; 0; 0], 0);
stats = validate_tile_placement(off, vtu, Geom);
assert(isequal(stats.NumTiles, 0));

%% Offsets that keep every rotation of the vessel inside the domain pass.
cfg = base; cfg.Transforms = [tile([0;0;0], 0), tile([0.002; 0.004; 0], 1.1)];
stats = validate_tile_placement(cfg, vtu, Geom);
assert(stats.MaxOverhang <= 0);
assert(isequal(stats.NumTiles, 2));

%% A tile pushed past the far wall of the domain is an error, not a warning.
cfg = base; cfg.Transforms = tile([0.008; 0; 0], 0);
threw = false;
try
    validate_tile_placement(cfg, vtu, Geom);
catch err
    threw = strcmp(err.identifier, 'generate_streamlines:TilePlacementOutsideDomain');
end
assert(threw);

%% With rotation on, the bound is the swept disc, because theta is random and
%% the check has to hold for every angle it can draw.
cfg = base; cfg.Transforms = tile([0; 0.017; 0], 0);   % 17 + 7.38 > 22.6 mm
threw = false;
try
    validate_tile_placement(cfg, vtu, Geom);
catch err
    threw = strcmp(err.identifier, 'generate_streamlines:TilePlacementOutsideDomain');
end
assert(threw);

%% With rotation off the vessel keeps its own footprint, so the same offset is
%% fine: 4 + 17 < 22.6 mm.
cfg = base; cfg.RandomizeRotation = false; cfg.Transforms = tile([0; 0.017; 0], 0);
stats = validate_tile_placement(cfg, vtu, Geom);
assert(stats.MaxOverhang <= 0);

%% Elevation is checked too, and it does not sweep -- rotation preserves it.
cfg = base; cfg.Transforms = tile([0; 0; 0.003], 0);   % 3 + 2 > 3.58 mm
threw = false;
try
    validate_tile_placement(cfg, vtu, Geom);
catch err
    threw = strcmp(err.identifier, 'generate_streamlines:TilePlacementOutsideDomain');
end
assert(threw);

disp('test_validate_tile_placement: all assertions passed');
