repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'streamline-module'));

% The real vessel->image rotation from the renal_tree configs, so the tests
% below exercise a frame where vessel and image axes genuinely differ.
R_geom = [0 0 -1; -1 0 0; 0 1 0];
Geom = struct( ...
    'Rotation', R_geom, ...
    'Center', [0.0342; 0; 0], ...
    'BoundingBox', struct('Center', [0.008; 0.007; 0.011]));

% Build a vessel point cloud from chosen image-frame positions:
%   image = R*(vessel - bb) + c   =>   vessel = R'*(image - c) + bb
img2vessel = @(img) (R_geom' * (img.' - Geom.Center) + Geom.BoundingBox.Center).';

elevations = 1e-3 * [-4 -1.2 -0.4 -0.05 0 0.05 0.4 1.2 4];
imgPts = [linspace(0.028, 0.040, numel(elevations)).', ...
          linspace(-0.004, 0.004, numel(elevations)).', ...
          elevations.'];
vtu = struct('points', img2vessel(imgPts), ...
             'density', ones(numel(elevations), 1) / numel(elevations));

%% Cells outside the slab lose their seeding weight; the rest keep theirs.
[cropped, stats] = crop_vessel_to_slab(vtu, Geom, 0.5e-3);
inSlab = abs(elevations(:)) <= 0.5e-3;
assert(all(cropped.density(~inSlab) == 0));
assert(all(cropped.density(inSlab) > 0));

%% The surviving weights are renormalized to a distribution.
assert(abs(sum(cropped.density) - 1) < 1e-12);

%% Relative weights among the survivors are untouched.
kept = find(inSlab);
assert(abs(cropped.density(kept(1)) - cropped.density(kept(end))) < 1e-12);

%% The stats say what was kept.
assert(isequal(stats.NTotal, numel(elevations)));
assert(isequal(stats.NInSlab, nnz(inSlab)));
assert(abs(stats.InSlabFraction - nnz(inSlab)/numel(elevations)) < 1e-12);
assert(isequal(stats.SeedableCells, nnz(inSlab)));
assert(abs(stats.HalfThickness - 0.5e-3) < 1e-15);

%% The slab is measured in the image frame, not in the vessel frame. Moving
%% the vessel bounding-box centre moves every cell's elevation by -(R*delta)_3
%% and so changes which cells survive; Geometry.Center cannot be used for this
%% check, since its elevation enters the transform and the slab centre alike
%% and cancels out.
delta = [0; 0.0012; 0];                     % row 3 of R is [0 1 0]
bbGeom = Geom;
bbGeom.BoundingBox.Center = Geom.BoundingBox.Center + delta;
[shifted, ~] = crop_vessel_to_slab(vtu, bbGeom, 0.5e-3);
expected = abs(elevations(:) - delta(2)) <= 0.5e-3;
assert(isequal(shifted.density > 0, expected));
assert(~isequal(expected, inSlab));         % the shift really changed the set

%% The crop commutes with the tile transform, which is why it may run before
%% tiling: a tile rotates about the elevation axis, so elevation is preserved.
theta = 0.9;
R_theta = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
R_v = R_geom' * R_theta * R_geom;
bb = Geom.BoundingBox.Center;
rotated = vtu;
rotated.points = (R_v * (vtu.points.' - bb) + bb).';
[rotatedCrop, ~] = crop_vessel_to_slab(rotated, Geom, 0.5e-3);
assert(isequal(rotatedCrop.density > 0, cropped.density > 0));

%% A slab that keeps nothing is an error, not a silent empty distribution.
% Every cell is moved 10 mm off the imaging plane, so a 0.5 mm slab is empty.
farVtu = vtu;
farVtu.points = img2vessel(imgPts + [0 0 0.01]);
threw = false;
try
    crop_vessel_to_slab(farVtu, Geom, 0.5e-3);
catch err
    threw = strcmp(err.identifier, 'crop_vessel_to_slab:empty');
end
assert(threw);

%% Seeding weights must exist first, exactly as crop_vessel_to_domain demands.
threw = false;
try
    crop_vessel_to_slab(rmfield(vtu, 'density'), Geom, 0.5e-3);
catch err
    threw = strcmp(err.identifier, 'crop_vessel_to_slab:noDensity');
end
assert(threw);

disp('test_crop_vessel_to_slab: all assertions passed');
