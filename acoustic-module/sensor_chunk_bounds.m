function [first, last] = sensor_chunk_bounds(N_rows, N_cols, useGPU, maxElements)
%SENSOR_CHUNK_BOUNDS Split an array axis into device-allocatable blocks.
%
%   [first, last] = SENSOR_CHUNK_BOUNDS(N_rows, N_cols, useGPU) returns the
%   first and last index of each block of an [N_rows x N_cols] array, such
%   that first(1) == 1, last(end) == N_rows and the blocks tile the range
%   without overlap.
%
%   A gpuArray holds at most intmax('int32') elements, whatever the device
%   has free, so any array whose two dimensions multiply past that has to be
%   held in pieces. Two axes are partitioned this way, both by this
%   function:
%
%     the sensor axis of the receive propagation's [N_sensor x Nt]
%     accumulator (run_simulation_homogeneous), which grows with the cube of
%     the grid refinement: v7 (lambda/6, 192 elements) needs
%     113963 x 9339 = 1.06e9 and fits, v10 (lambda/8) needs
%     202140 x 12453 = 2.52e9 and does not;
%
%     the element axis of the receive processing's [N_el*N_int x N] work
%     (compute_RF), where a row costs N_int*N elements rather than one, so
%     N_cols is passed as N_int*N.
%
%   Either split costs nothing in physics: a sensor row depends on no other
%   sensor row, and an element's RF line depends on no other element.
%
%   CPU arrays have no such limit, so useGPU == false always returns one
%   block. Passing maxElements overrides both the limit and that shortcut,
%   which is how the split path is exercised without a GPU, and how a caller
%   bounds working memory rather than only the element count.
%
%   Blocks are balanced rather than greedy: a greedy split of v10's sensor
%   axis would give 172447 and 29693 rows, whose peak is the 172447 one,
%   where two even blocks peak at 101070.

if nargin < 4 || isempty(maxElements)
    if ~useGPU
        first = 1;
        last  = N_rows;
        return
    end
    maxElements = double(intmax('int32'));
end

if N_rows * N_cols <= maxElements
    first = 1;
    last  = N_rows;
    return
end

% One row wider than the limit cannot be split any further. Returning
% single rows is the closest this can get; the caller sees the device error
% rather than a silently wrong answer.
rowsPerChunk = max(1, floor(maxElements / N_cols));

N_chunk = ceil(N_rows / rowsPerChunk);
rowsPerChunk = ceil(N_rows / N_chunk);

first = 1:rowsPerChunk:N_rows;
last  = min(first + rowsPerChunk - 1, N_rows);

end
