% Every per-bubble field must lose the same rows when a bubble falls off the
% grid. Until 2026-09-03 only radii and velocities were pruned, so TileID and
% RawVelocity kept their pre-exclusion length and, from the first dropped
% bubble on, described a different bubble than Points did. main_RF saves this
% struct into the RF frame, so the mislabelling reached the data.

clear MBx Gridx sensorx weightsx keptx n_inx

Gridx = struct('Nx',40,'Ny',38,'Nz',36,'dx',1e-4,'dy',1e-4,'dz',1e-4, ...
               'sensor_on_grid',false);
Gridx.x = (0:Gridx.Nx-1)*Gridx.dx;
Gridx.y = (0:Gridx.Ny-1)*Gridx.dy;
Gridx.z = (0:Gridx.Nz-1)*Gridx.dz;
Gridx.full_size = [Gridx.Nx; Gridx.Ny; Gridx.Nz];

% Five bubbles: three inside the grid, two outside it (one below the x
% minimum, one above the z maximum).
MBx = struct();
MBx.points = [ ...
    Gridx.x(12) Gridx.y(14) Gridx.z(16); ...
    -5*Gridx.dx Gridx.y(14) Gridx.z(16); ...   % outside
    Gridx.x(20) Gridx.y(18) Gridx.z(20); ...
    Gridx.x(20) Gridx.y(18) 99*Gridx.dz; ...   % outside
    Gridx.x(24) Gridx.y(22) Gridx.z(24)];
MBx.radii          = (1:5)' * 1e-6;
MBx.velocities     = repmat((1:5)', 1, 3);
MBx.raw_velocities = repmat((11:15)', 1, 3);
MBx.tile_ids       = (101:105)';

n_inx = size(MBx.points, 1);
[sensorx, weightsx, MBx] = define_sensor_MB(Gridx, MBx);

keptx = [1 3 5];
assert(size(MBx.points,1) == 3, 'two bubbles should have been dropped')
assert(isequal(MBx.radii,          (keptx').*1e-6),        'radii not pruned')
assert(isequal(MBx.velocities,     repmat(keptx',1,3)),    'velocities not pruned')
assert(isequal(MBx.raw_velocities, repmat(keptx'+10,1,3)), 'raw_velocities not pruned')
assert(isequal(MBx.tile_ids,       keptx'+100),            'tile_ids not pruned')

% Every per-bubble field now agrees with points, which is the invariant the
% saved Frame relies on.
assert(numel(MBx.radii)    == size(MBx.points,1))
assert(numel(MBx.tile_ids) == size(MBx.points,1))
assert(size(MBx.raw_velocities,1) == size(MBx.points,1))

% The sensor built from the kept bubbles is unaffected by the pruning.
assert(size(weightsx,1) == 3, 'one weight row per kept bubble')
assert(nnz(sensorx.mask) > 0)

% A frame where nothing is excluded must come through untouched.
clear MBx
MBx = struct('points', [Gridx.x(12) Gridx.y(14) Gridx.z(16)], ...
             'radii', 1e-6, 'velocities', [1 1 1], ...
             'raw_velocities', [2 2 2], 'tile_ids', 7);
[~, ~, MBx] = define_sensor_MB(Gridx, MBx);
assert(isequal(MBx.tile_ids, 7))
assert(size(MBx.points,1) == 1)

disp('test_define_sensor_MB_exclusion: all assertions passed')
