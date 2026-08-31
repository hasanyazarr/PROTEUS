function [first, last] = rf_sensor_chunks(N_sensor, M, useGPU, run_param)
%RF_SENSOR_CHUNKS Split the sensor axis of the recorded pressure.
%
%   Bounds both the upload and the double copy the sparse product makes of
%   each chunk: MATLAB has no single-precision sparse, so sensor_weights
%   forces a double operand, and a double copy of the whole record is twice
%   the array that did not fit in the first place.

CHUNK_BYTES = 2*2^30;
BYTES_PER_DOUBLE = 8;

if isfield(run_param, 'RFSensorChunkElements') && ...
        ~isempty(run_param.RFSensorChunkElements)
    maxElements = run_param.RFSensorChunkElements;   % tests only
else
    maxElements = min(double(intmax('int32')), ...
        floor(CHUNK_BYTES/BYTES_PER_DOUBLE));
end

[first, last] = sensor_chunk_bounds(N_sensor, M, useGPU, maxElements);

end
