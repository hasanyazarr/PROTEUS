function [first, last] = prop_sensor_chunks(N_sensor, Nt, dataType, ...
    useGPU, run_param)
%PROP_SENSOR_CHUNKS Split the sensor axis of the receive-propagation accumulator.
%
%   The gpuArray element cap is what forces a split at all, but it is not
%   what should size one. run_simulation_homogeneous expands the distance
%   grid back to a whole chunk before adding it -- p_sensor, plus the chunk
%   the add itself reads -- so the transient is about two chunks, and it is
%   rebuilt once per source. Sizing chunks by intmax('int32') alone made
%   that 2 x 4.69 GiB at v11's grid, on top of the 9.38 GiB of accumulators
%   that stay resident either way.
%
%   Splitting more finely costs no arithmetic: every sensor row is touched
%   exactly once per source whatever the split, so only the kernel-launch
%   count changes. This mirrors the byte budgets already in
%   rf_sensor_chunks and rf_element_blocks -- the propagation path was the
%   one place still partitioning on the element cap alone.
%
%   What this does NOT do: the accumulators are all allocated before the
%   source loop, so a smaller chunk means more resident chunks summing to
%   the same total. Bounding the resident total needs the sensor block
%   moved outside the source loop, which recomputes the propagation per
%   block -- a different change, with a real cost.

CHUNK_BYTES = 2^30;

if strcmp(dataType, 'double')
    bytesPerElement = 8;
else
    bytesPerElement = 4;
end

if isfield(run_param, 'SensorChunkElements') && ...
        ~isempty(run_param.SensorChunkElements)
    maxElements = run_param.SensorChunkElements;   % tests only
else
    maxElements = min(double(intmax('int32')), ...
        floor(CHUNK_BYTES/bytesPerElement));
end

[first, last] = sensor_chunk_bounds(N_sensor, Nt, useGPU, maxElements);

end
