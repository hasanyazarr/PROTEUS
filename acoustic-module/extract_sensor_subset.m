function sensor_subset = extract_sensor_subset( ...
    sensor_data, source_mask_idx, target_mask_idx, n_time_points)
%EXTRACT_SENSOR_SUBSET Take one sensor set's rows out of a combined record.
%
%   The combined transmit runs one k-Wave simulation over the union of the
%   transducer and microbubble masks, then splits it into the two records the
%   split transmit would have produced separately. This is that split.
%
%   It rests on how k-Wave orders a record: sensor points come back in linear
%   index order, so a run over one mask returns find(mask)'s rows ascending.
%   SOURCE_MASK_IDX and TARGET_MASK_IDX are both find() output and both
%   ascending, so intersect's positions into the source are ascending too --
%   which is the order the target's own run would have used.
%
%   N_TIME_POINTS is where the microbubble rows lose the round trip they were
%   recorded over. The bubbles only ever read the one-way window.
%
%   Extracted from main_RF's local functions on 2026-09-05 so that
%   tests/matlab/test_combined_split_extraction_equivalence.m can reach it.
%   The preflight now moves runs between the two paths on its own, so the
%   claim that they agree is one the code relies on rather than one a person
%   makes per run.

[common, sensor_data_idx] = intersect(source_mask_idx, target_mask_idx);

% intersect would otherwise hand back the rows it did find, and a record
% short of rows still multiplies against a projection built for all of them
% in shapes forgiving enough to go unnoticed.
if numel(common) ~= numel(target_mask_idx)
    error('extract_sensor_subset:TargetNotRecorded', ...
        ['%d of the %d requested sensor points are not in the record. ' ...
         'The record must be made over a mask containing the target''s.'], ...
        numel(target_mask_idx) - numel(common), numel(target_mask_idx));
end

sensor_subset.p = sensor_data.p(sensor_data_idx, 1:n_time_points);

end
