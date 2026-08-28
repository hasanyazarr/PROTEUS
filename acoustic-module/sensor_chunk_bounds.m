function [first, last] = sensor_chunk_bounds(N_sensor, Nt, useGPU, maxElements)
%SENSOR_CHUNK_BOUNDS Split the sensor axis into device-allocatable blocks.
%
%   [first, last] = SENSOR_CHUNK_BOUNDS(N_sensor, Nt, useGPU) returns the
%   first and last sensor index of each chunk of an [N_sensor x Nt]
%   accumulator, such that first(1) == 1, last(end) == N_sensor and the
%   chunks tile the range without overlap.
%
%   A gpuArray holds at most intmax('int32') elements, whatever the device
%   has free. The receive propagation accumulates one row per transducer
%   grid point and one column per receive sample, so the accumulator grows
%   with the cube of the grid refinement: v7 (lambda/6, 192 elements) needs
%   113963 x 9339 = 1.06e9 and fits, v10 (lambda/8) needs 202140 x 12453 =
%   2.52e9 and does not. Splitting the sensor axis keeps every array under
%   the limit; it costs nothing in physics, because a sensor row depends on
%   no other sensor row.
%
%   CPU arrays have no such limit, so useGPU == false always returns one
%   chunk. Passing maxElements overrides both the limit and that shortcut,
%   which is how the chunked path is exercised without a GPU.
%
%   Chunks are balanced rather than greedy: a greedy split of v10 would give
%   172447 and 29693 rows, whose peak is the 172447 one, where two even
%   chunks peak at 101070.

if nargin < 4 || isempty(maxElements)
    if ~useGPU
        first = 1;
        last  = N_sensor;
        return
    end
    maxElements = double(intmax('int32'));
end

if N_sensor * Nt <= maxElements
    first = 1;
    last  = N_sensor;
    return
end

% One row wider than the limit cannot be split any further. Returning
% single rows is the closest this can get; the caller sees the device error
% rather than a silently wrong answer.
rowsPerChunk = max(1, floor(maxElements / Nt));

N_chunk = ceil(N_sensor / rowsPerChunk);
rowsPerChunk = ceil(N_sensor / N_chunk);

first = 1:rowsPerChunk:N_sensor;
last  = min(first + rowsPerChunk - 1, N_sensor);

end
