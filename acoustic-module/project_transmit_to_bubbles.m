function sensed = project_transmit_to_bubbles( ...
    record, W, n_time, dataType, budget_bytes)
%PROJECT_TRANSMIT_TO_BUBBLES Stream a recorded transmit onto the bubbles.
%
%   sensed = PROJECT_TRANSMIT_TO_BUBBLES(RECORD, W, N_TIME, DATATYPE)
%   returns W * p over the first N_TIME samples of the recorded pressure p,
%   without ever holding p whole.
%
%   RECORD is either the name of a k-Wave output file, whose '/p' dataset is
%   read in blocks, or the record itself as a matrix, for the combined
%   transmit path which already has it in memory. Both go through the same
%   blocked product, so neither doubles its input to cast it.
%
%   W has one row per (frame, pulse, bubble) and one column per point of the
%   batch's sensor mask, so the product is the pressure each bubble senses.
%   It is the same product the frame loop used to do one frame at a time:
%
%       rows     = positions of this frame's mask inside the batch mask
%       sensed_p = sensor_weights_frame * double(p(rows, 1:n_time))
%
%   Writing each frame's weights into the batch mask's columns and stacking
%   the frames gives this W, and W * p is that product for every frame at
%   once -- algebraically the same numbers, not an approximation. What
%   changes is that p is read in time blocks and discarded, so the peak is
%   one block rather than the whole record: 281 GB at v11's grid, against
%   83 GB of host memory.
%
%   Doing it this way also removes the per-frame intersect and sparse
%   product from the frame loop, measured together at 2.4 s of a 14.8 s
%   frame.
%
%   BUDGET_BYTES caps the working copy of one block. Default 2 GB.

if nargin < 5 || isempty(budget_bytes)
    budget_bytes = 2 * 2^30;
end

n_rows = size(W, 2);
n_out  = size(W, 1);

from_file = ischar(record) || isstring(record);

% The record's shape, checked rather than assumed. k-Wave stores the sampled
% pressure with the sensor index fastest, so h5read hands back
% [N_sensor x Nt] and a time block is a contiguous read. If a toolbox
% version ever changed that, every block below would be silently wrong, so
% it is an error rather than a comment.
if from_file
    info = h5info(record, '/p');
    dims = info.Dataspace.Size;
else
    dims = size(record);
end
if numel(dims) < 2 || dims(1) ~= n_rows
    error('project_transmit_to_bubbles:UnexpectedLayout', ...
        ['The record is %s but the sensor mask has %d points. This code ' ...
         'reads it as [N_sensor x Nt] and blocks over time.'], ...
        mat2str(dims), n_rows);
end
n_available = dims(2);
if n_time > n_available
    error('project_transmit_to_bubbles:RecordTooShort', ...
        '/p holds %d samples, %d were asked for.', n_available, n_time);
end

% One block, cast to double for the sparse product, is what has to fit.
block_cols = max(1, floor(budget_bytes / (n_rows * 8)));
block_cols = min(block_cols, n_time);
n_blocks   = ceil(n_time / block_cols);

run_log('banner', 'project', ...
    ['Projecting %d x %d transmit onto %d bubble rows in %d block(s) ' ...
     'of <=%d samples'], n_rows, n_time, n_out, n_blocks, block_cols);

sensed = zeros(n_out, n_time, dataType);

for c0 = 1:block_cols:n_time
    c1 = min(c0 + block_cols - 1, n_time);
    if from_file
        block = h5read(record, '/p', ...
            [1 c0 ones(1, numel(dims) - 2)], ...
            [n_rows (c1 - c0 + 1) ones(1, numel(dims) - 2)]);
        block = reshape(block, n_rows, []);
    else
        block = record(:, c0:c1);
    end
    % MATLAB has no single-precision sparse, so the product goes through
    % double either way. This is the same cast the frame loop did.
    sensed(:, c0:c1) = cast(full(W * double(block)), dataType);
end

end
