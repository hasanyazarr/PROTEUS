% Chunking the sensor axis must not change the answer.
%
% The GPU path cannot run here, so run_param.SensorChunkElements forces the
% same split on the CPU and the two results are compared bit for bit. If they
% ever differ, the chunk loop has stopped being a pure re-slicing of the
% accumulation.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));
run_log('reset');

rng(7);

N_source = 6;
N_sensor = 97;              % deliberately not a multiple of any chunk size
Nt = 64;

kgrid.dt = 1/50e6;
kgrid.Nt = Nt;

medium.density     = 1000 * ones(4,4,4);
medium.sound_speed = 1540 * ones(4,4,4);
medium.alpha_coeff = 0.75 * ones(4,4,4);
medium.alpha_power = 1.5  * ones(4,4,4);
medium.alpha_mode  = 'no_dispersion';   % keeps k-Wave off the path

source.points      = 1e-3 * randn(N_source, 3);
source.mass_source = single(randn(N_source, Nt));
sensor.points      = 1e-2 * randn(N_sensor, 3) + [0.02 0 0];

% A sensor sitting exactly on a source, to keep the self-sensing mask live.
sensor.points(13,:) = source.points(2,:);

run_param.DATA_CAST  = 'single';
run_param.DEVICE_NUM = 0;

for gridded = [false true]
    run_param.gridded = gridded;

    run_param.SensorChunkElements = [];        % one chunk
    whole = run_simulation_homogeneous(run_param, kgrid, medium, source, sensor);

    for limit = [N_sensor*Nt, 40*Nt, 10*Nt, Nt]
        run_param.SensorChunkElements = limit;
        [f, l] = sensor_chunk_bounds(N_sensor, Nt, false, limit);
        split = run_simulation_homogeneous( ...
            run_param, kgrid, medium, source, sensor);

        assert(isequal(size(split.p), [N_sensor Nt]), ...
            'chunked result changed shape');
        assert(isequal(class(split.p), class(whole.p)), ...
            'chunked result changed class');
        assert(isequal(split.p, whole.p), ...
            sprintf('gridded=%d limit=%d: %d chunks changed the result', ...
                    gridded, limit, numel(f)));
        assert(l(end) == N_sensor);
    end
end

% The self-sensing row really is zeroed, so the comparison above is not
% comparing two equally broken masks. One source only: with several, the
% other five still reach sensor 13.
run_param.gridded = false;
run_param.SensorChunkElements = [];
lone.points      = source.points(2,:);
lone.mass_source = source.mass_source(2,:);
out = run_simulation_homogeneous(run_param, kgrid, medium, lone, sensor);
assert(all(out.p(13,:) == 0), 'self-sensing row is not zero');
assert(any(out.p(1,:) ~= 0), 'nothing propagated -- the test is vacuous');

% And the mask survives the split: sensor 13 lands in a later chunk here.
run_param.SensorChunkElements = 10*Nt;
out_split = run_simulation_homogeneous(run_param, kgrid, medium, lone, sensor);
assert(isequal(out_split.p, out.p), 'chunking moved the self-sensing mask');

disp('test_run_simulation_homogeneous_chunking: all assertions passed');
