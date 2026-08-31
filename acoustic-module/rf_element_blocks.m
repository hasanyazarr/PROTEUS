function [first, last] = rf_element_blocks(N_el, N_int, N, dataType, ...
    useGPU, run_param)
%RF_ELEMENT_BLOCKS Split the element axis so one [n_el*N_int x N] fits.
%
%   Two of those intermediates are live at once in compute_RF -- the
%   transform and the delay it is multiplied by -- and both are complex, so
%   the byte budget below is per array and the working set is about twice
%   it. The gpuArray element limit applies on top: a block has to satisfy
%   both, which is what taking the minimum below does.
%
%   The budget bounds the host path as well as the device one. The CPU
%   retry in compute_RF_data exists to survive a device that cannot do the
%   work, and an unblocked host path would only trade a device limit for
%   38 GB of host allocation.

BLOCK_BYTES = 2*2^30;

if strcmp(dataType,'double')
    bytesPerElement = 16;   % complex double
else
    bytesPerElement = 8;    % complex single
end

if isfield(run_param, 'RFBlockElements') && ~isempty(run_param.RFBlockElements)
    maxElements = run_param.RFBlockElements;   % tests only
else
    maxElements = min(double(intmax('int32')), ...
        floor(BLOCK_BYTES/bytesPerElement));
end

% Each element costs N_int*N elements of the intermediate rather than one,
% so the element axis partitions exactly as the sensor axis does.
[first, last] = sensor_chunk_bounds(N_el, N_int*N, useGPU, maxElements);

end
