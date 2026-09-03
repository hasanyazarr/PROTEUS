% =========================================================================
% RUN THE SIMULATION
% input:    run_param
%           PML 
%           kgrid
%           pulse
%           medium
%           source
%           sensor
% output:   sensor_data - output of kWave simulation function 
%           kspaceFirstOrder, contains recorded pressure lines on sensors
% 
% According to run_param structure the kWave simulation function is chosen
% for gpu - kspaceFirstOrder3DG, for cpu - kspaceFirstOrder3DC, local
% machine - kspaceFirstOrder3D. To run simulation on gpu CUDA module is
% necessary. 
% =========================================================================
function sensor_data = run_simulation(...
    run_param, kgrid, medium, source, sensor)

% Shared with run_simulation_to_disk, which runs the same simulation and
% leaves its output on disk. The two must not drift apart.
input_args = kwave_input_args(run_param);

% DeleteData only supported by 3DC/3DG (C binary); kspaceFirstOrder3D rejects it
if isfield(run_param,'BINARY_PATH')
    input_args = [input_args {'DeleteData', true}];  % Delete h5 files after each frame to save disk
end

if isfield(run_param,'DATA_PATH') 
    input_args = [input_args {'DataPath', run_param.DATA_PATH}];
end
    
if isfield(run_param,'DEVICE_NUM')
    input_args = [input_args {'DeviceNum', run_param.DEVICE_NUM}];
end

if isfield(run_param,'BINARY_PATH')
    input_args = [input_args {'BinaryPath', run_param.BINARY_PATH}];
end
    

if run_param.record_movie && isfield(source,'u_mask')
    
    rho = mean(medium.density(:));
    c   = mean(medium.sound_speed(:));
    
    input_args = [input_args {...
        'DisplayMask',  source.u_mask, ...
        'PlotPML',      false, ...
        'PlotScale',    [-1/4, 1/4] *  max(source.ux*rho*c, [], 'all'), ...
        'RecordMovie',  true, ...
        'MovieName',    'movie', ...
        'MovieProfile', 'MPEG-4'}];
    
elseif run_param.record_movie
    
	input_args = [input_args {...
        'DisplayMask',  sensor.mask, ...
        'PlotPML',      false, ...
        'PlotScale',    [-1/4, 1/4] *  max(source.p, [], 'all'), ...
        'RecordMovie',  true, ...
        'MovieName',    'movie', ...
        'MovieProfile', 'MPEG-4'}];

end

% If no sensors points in sensor mask, return empty array:
if sum(sensor.mask,'all') == 0
    sensor_data.p = zeros(0,kgrid.Nt);
    warning(['No nonzero elements in sensor mask, ' ...
        'returning empty sensor data.'])
    return
end

sensor_data = feval(run_param.solver, kgrid, medium, source, sensor, ...
    input_args{:});


end