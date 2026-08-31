function [dataType, useGPU] = rf_data_cast(run_param)
%RF_DATA_CAST Read the receive path's precision and device from run_param.
%
%   Shared by compute_RF and preflight_array_limits so the preflight sizes
%   the arrays the run will actually build, in the precision it will build
%   them in.

switch run_param.DATA_CAST_RF
    case 'gpuArray-single'
        dataType = 'single';
        useGPU   = true;
    case 'gpuArray-double'
        dataType = 'double';
        useGPU   = true;
    case 'single'
        dataType = 'single';
        useGPU   = false;
    case 'double'
        dataType = 'double';
        useGPU   = false;
    otherwise
        error('PROTEUS:rf_data_cast:unknownCast', ...
            'Unknown DATA_CAST_RF: %s', run_param.DATA_CAST_RF);
end

end
