repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));

LIMIT = double(intmax('int32'));  % elements in one gpuArray

%% A CPU accumulator is never split -- the element limit is the device's.
[first, last] = sensor_chunk_bounds(202140, 12453, false);
assert(isequal(first, 1));
assert(isequal(last, 202140));

%% A GPU accumulator that fits stays in one piece.
[first, last] = sensor_chunk_bounds(113963, 9339, true);   % v7 production
assert(113963 * 9339 <= LIMIT);
assert(isequal(first, 1));
assert(isequal(last, 113963));

%% v10 (lambda/8, 192 elements) is over the limit and gets split.
N_sensor = 202140;
Nt = 12453;
assert(N_sensor * Nt > LIMIT);
[first, last] = sensor_chunk_bounds(N_sensor, Nt, true);
assert(numel(first) > 1);
check_cover(first, last, N_sensor, Nt, LIMIT);

%% Chunks are balanced: no chunk is more than one row longer than another.
rows = last - first + 1;
assert(max(rows) - min(rows) <= 1);

%% A range of shapes stays a partition with every chunk under the limit.
for N_sensor = [1, 2, 12345, 202140, 700000, 3000000]
    for Nt = [1, 100, 9339, 12453, 40000]
        [first, last] = sensor_chunk_bounds(N_sensor, Nt, true);
        check_cover(first, last, N_sensor, Nt, LIMIT);
    end
end

%% A single row wider than the limit cannot be split further; it is not an
%% error here, it is the caller's problem to notice.
Nt = LIMIT + 10;
[first, last] = sensor_chunk_bounds(3, Nt, true);
assert(isequal(first(:)', 1:3));
assert(isequal(last(:)', 1:3));

disp('test_sensor_chunk_bounds: all assertions passed');

function check_cover(first, last, N_sensor, Nt, LIMIT)
% The chunks must tile 1:N_sensor exactly once, in order.
assert(numel(first) == numel(last));
assert(first(1) == 1);
assert(last(end) == N_sensor);
assert(all(last >= first));
assert(isequal(first(2:end), last(1:end-1) + 1));
rows = last - first + 1;
% Every chunk fits, unless one row alone already does not.
assert(all(rows * Nt <= LIMIT) || Nt > LIMIT);
end
