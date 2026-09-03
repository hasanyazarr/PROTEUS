function Projection = build_bubble_projection( ...
    Grid, folder, Acquisition, N_sequence, Geometry, mask_idx_batch, th)
%BUILD_BUBBLE_PROJECTION One matrix per pulse taking a transmit to bubbles.
%
%   Projection(pulse).W        [n_rows x numel(mask_idx_batch)] sparse
%   Projection(pulse).RowFirst [n_frames x 1]  first row of each frame
%   Projection(pulse).RowLast  [n_frames x 1]  last row of each frame
%   Projection(pulse).MB       {n_frames x 1}  the frame's bubbles
%   Projection(pulse).MaxDist  [n_frames x 1]  max pairwise bubble distance
%
%   The frame loop used to read one frame's rows out of the cached transmit
%   and multiply them by that frame's weights. Stacking every frame's
%   weights into one matrix, addressed against the batch's mask instead of
%   the frame's own, turns that into a single product the whole recorded
%   transmit can be streamed through -- see project_transmit_to_bubbles for
%   why the cache had to go.
%
%   The bubbles and their pairwise extent are carried along because this
%   pass already computes them; the frame loop would otherwise load and
%   voxelise every frame a second time. It is a few megabytes for a whole
%   acquisition.

n_frames = Acquisition.EndFrame - Acquisition.StartFrame + 1;
n_union  = numel(mask_idx_batch);

Projection = struct('W', cell(1, N_sequence), 'RowFirst', [], ...
    'RowLast', [], 'MB', [], 'MaxDist', []);

for pulse_seq_idx = 1:N_sequence

    rows_i = cell(n_frames, 1);
    cols_j = cell(n_frames, 1);
    vals   = cell(n_frames, 1);
    MB_all = cell(n_frames, 1);

    row_first = zeros(n_frames, 1);
    row_last  = zeros(n_frames, 1);
    max_dist  = zeros(n_frames, 1);
    next_row  = 1;

    for frame = Acquisition.StartFrame : Acquisition.EndFrame

        f = frame - Acquisition.StartFrame + 1;

        MB = load_microbubbles(folder, frame, pulse_seq_idx, Geometry, ...
            Acquisition.NumberOfFrames);

        % The frame's own sensor, exactly as the frame loop built it.
        [sensor_frame, weights_frame, MB, max_dist(f)] = ...
            define_sensor_MB(Grid, MB, th);

        % Relocate the weights from this frame's mask onto the batch's. The
        % frame's mask is a subset of the batch union by construction --
        % define_sensor_MB_all walked these same frames to build it -- so
        % this is a lookup, and locate_in_sorted errors if it ever is not.
        mask_idx_frame = find(sensor_frame.mask);
        cols = locate_in_sorted(mask_idx_batch, mask_idx_frame);

        [i, j, v] = find(weights_frame);
        n_bubbles = size(weights_frame, 1);

        rows_i{f} = i(:) + (next_row - 1);
        cols_j{f} = cols(j(:));
        vals{f}   = v(:);

        row_first(f) = next_row;
        row_last(f)  = next_row + n_bubbles - 1;
        next_row     = next_row + n_bubbles;

        MB_all{f} = MB;

        if mod(f, 25) == 0 || f == n_frames
            run_log('banner', 'projection', ...
                'Pulse %d: weights built for %d/%d frames', ...
                pulse_seq_idx, f, n_frames);
        end

    end

    Projection(pulse_seq_idx).W = sparse( ...
        vertcat(rows_i{:}), vertcat(cols_j{:}), vertcat(vals{:}), ...
        next_row - 1, n_union);
    Projection(pulse_seq_idx).RowFirst = row_first;
    Projection(pulse_seq_idx).RowLast  = row_last;
    Projection(pulse_seq_idx).MB       = MB_all;
    Projection(pulse_seq_idx).MaxDist  = max_dist;

end

end
