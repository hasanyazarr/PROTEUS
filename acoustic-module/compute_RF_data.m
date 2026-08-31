function [RF, run_param] = compute_RF_data(Transducer, sensor_data, ...
    sensor_weights, Grid, run_param)
%COMPUTE_RF_DATA Wrapper function for compute_RF. 
% If run_param.DATA_CAST_RF is of GPU type, but compute_RF cannot complete
% on the device, run_param.DATA_CAST_RF is changed to CPU type and the
% computation is retried on the host.
%
% The retry covers any parallel:gpu:* failure, not out-of-memory alone. The
% v10 run died here on 30-Aug-2026 with
%
%   Error using gpuArray
%   Maximum variable size allowed on the device is exceeded.
%
% which is a size-limit failure rather than an OOM, so the identifier did
% not match 'parallel:gpu:array:OOM' and the exception was rethrown after
% two hours of transmit simulation. Every parallel:gpu:* failure describes a
% device that cannot do the work, and the host can, so the prefix is the
% right test; anything else still rethrows. compute_RF blocks its arrays on
% the host as well, so the retry is bounded in memory rather than trading a
% device limit for a host one.
%
% Nathan Blanken, University of Twente, 2023

try
    RF = compute_RF(Transducer,sensor_data,sensor_weights,Grid,run_param);
catch ME
    if ~startsWith(ME.identifier, 'parallel:gpu:')
        rethrow(ME)
    end

    disp(['WARNING: the GPU could not complete the RF computation: ' ...
        ME.message]);
    disp('Switching to CPU.');
    
    % Change the data cast for the RF data from GPU to CPU:
    switch run_param.DATA_CAST_RF
        case 'gpuArray-double'
            run_param.DATA_CAST_RF = 'double';
        case 'gpuArray-single'
            run_param.DATA_CAST_RF = 'single';
    end

    % Retry the RF computation on the CPU:
    RF = compute_RF(Transducer,sensor_data,sensor_weights,Grid,run_param);
end

end
