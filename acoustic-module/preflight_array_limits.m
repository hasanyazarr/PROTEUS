function preflight_array_limits(Transducer, N_sensor, M, N_pulses, ...
    Grid, run_param)
%PREFLIGHT_ARRAY_LIMITS Refuse a run whose receive arrays cannot be held.
%
%   Called before the first k-Wave transmit, from quantities all known by
%   then, so a configuration that cannot complete says so in seconds. The
%   v10 500-frame run learned the same thing from the exception in
%   compute_RF after two hours of transmit simulation and zero frames
%   written.
%
%   Every array the receive path builds is sized by the transducer's
%   integration points and the receive window, and on the GPU each one is a
%   single gpuArray, capped at intmax('int32') elements whatever the device
%   has free. compute_RF blocks the element axis and chunks the sensor axis
%   to stay under that cap; this function checks that the blocking is
%   actually enough, which it is not once a single element's own work
%   crosses the limit. At that point no code change helps -- the grid
%   refinement, the element height or the integration density has to give --
%   so this errors rather than letting the run start.
%
%   Host memory is reported, not enforced: MATLAB on Linux cannot query the
%   machine's free memory, so the number is for the reader.

limit = double(intmax('int32'));

[dataType, useGPU] = rf_data_cast(run_param);
[N_el, N_int, ~]   = size(Transducer.integration_points);
N = rf_signal_length(Transducer, Grid, M, dataType);

[el_first, el_last] = rf_element_blocks(N_el, N_int, N, dataType, ...
    useGPU, run_param);
[s_first, s_last]   = rf_sensor_chunks(N_sensor, M, useGPU, run_param);

N_block = numel(el_first);
N_chunk = numel(s_first);

% The largest array each loop actually allocates.
blockRows  = max(el_last - el_first + 1) * N_int;
chunkRows  = max(s_last - s_first + 1);
worstBlock = blockRows * N;
worstChunk = chunkRows * M;

fprintf('=== Receive-path preflight ===\n');
fprintf('  %-28s %11s %11s %s\n', 'array', 'whole', 'blocked', 'shape');
report('recorded pressure', N_sensor*M, worstChunk, ...
    sprintf('%d x %d -> %d chunk(s)', N_sensor, M, N_chunk));
report('integration-point record', N_el*N_int*M, blockRows*M, ...
    sprintf('%d x %d -> %d block(s)', N_el*N_int, M, N_block));
report('delayed spectrum (complex)', N_el*N_int*N, worstBlock, ...
    sprintf('%d x %d -> %d block(s)', N_el*N_int, N, N_block));

bytesPerSample = 4;
if strcmp(dataType,'double')
    bytesPerSample = 8;
end
record_GB = N_sensor * M * bytesPerSample / 2^30;
fprintf(['  host: transmit cache %d x %.1f GB, plus about 2 x %.1f GB ' ...
    'per frame\n'], N_pulses, record_GB, record_GB);

if useGPU && (worstBlock > limit || worstChunk > limit)
    error('PROTEUS:preflight:arrayOverLimit', ...
        ['The receive path cannot be held on the device even blocked: ' ...
         'largest block is %.3e elements against the %.3e gpuArray ' ...
         'limit. One element already needs %d x %d = %.3e. Reduce the ' ...
         'grid refinement (PointsPerWavelength), the element height, or ' ...
         'IntegrationDensity -- no blocking fixes this.'], ...
        max(worstBlock, worstChunk), limit, N_int, N, N_int*N);
end

    function report(name, whole, blocked, shape)
        mark = '';
        if whole > limit
            mark = '  over the limit whole';
        end
        fprintf('  %-28s %11.3e %11.3e %s%s\n', ...
            name, whole, blocked, shape, mark);
    end

end
