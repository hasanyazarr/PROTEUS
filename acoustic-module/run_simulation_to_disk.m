function output_filename = run_simulation_to_disk(...
    run_param, kgrid, medium, source, sensor)
%RUN_SIMULATION_TO_DISK Run a k-Wave simulation, return the output file.
%
%   Same simulation as run_simulation, but the recorded pressure is left in
%   the binary's HDF5 output instead of being read into a MATLAB array.
%
%   Why this exists: the microbubble transmit is recorded at the union of
%   every bubble position over the batch, and with tiling that union is 54x
%   the transducer mask. run_simulation ends in
%
%       sensor_data.p = h5read(output_filename, '/p')
%
%   inside kspaceFirstOrder3DC, which materialises the whole record -- 281 GB
%   at v11's grid, against 83 GB of host memory. The frame loop never wants
%   the whole record, only one frame's rows projected onto its bubbles, so
%   the caller streams the file instead. See project_transmit_to_bubbles.
%
%   The caller owns both files and must delete them; nothing here passes
%   'DeleteData', because the output is the return value.

if ~isfield(run_param, 'BINARY_PATH')
    error('run_simulation_to_disk:NoBinary', ...
        ['Streaming the transmit needs the C/CUDA binary. Solver %s runs ' ...
         'in MATLAB and returns its data in memory.'], run_param.solver);
end

if sum(sensor.mask, 'all') == 0
    error('run_simulation_to_disk:EmptySensor', ...
        'No nonzero elements in the sensor mask; there is nothing to record.')
end

% ------------------------------------------------------------------ files
data_path = run_param.DATA_PATH;
if ~strcmp(data_path(end), filesep)
    data_path = [data_path filesep];
end
if ~exist(data_path, 'dir')
    mkdir(data_path);
end
stamp = datestr(now, 'dd-mmm-yyyy-HH-MM-SS'); %#ok<TNOW1,DATST>
input_filename  = [data_path 'kwave_stream_input'  stamp '.h5'];
output_filename = [data_path 'kwave_stream_output' stamp '.h5'];

% --------------------------------------------------------- write the input
% kspaceFirstOrder3D with SaveToDisk writes the input file and returns
% without simulating. This is the same call kspaceFirstOrder3DC makes, with
% the same options, which is why they come from a shared helper.
input_args = kwave_input_args(run_param);
kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:}, ...
    'SaveToDisk', input_filename);

% ---------------------------------------------------------- run the binary
binary_path = run_param.BINARY_PATH;
if ~strcmp(binary_path(end), filesep)
    binary_path = [binary_path filesep];
end
if isfield(run_param, 'BINARY_NAME')
    binary_name = run_param.BINARY_NAME;
elseif strcmp(run_param.solver, 'kspaceFirstOrder3DG')
    binary_name = 'kspaceFirstOrder-CUDA';
else
    binary_name = 'kspaceFirstOrder-OMP';
end
if ~exist([binary_path binary_name], 'file')
    delete(input_filename);
    error('run_simulation_to_disk:BinaryNotFound', ...
        'Could not find %s in %s', binary_name, binary_path);
end

% These sensors only ever record p; sensor.record is set to {'p'} by
% define_sensor_MB_all and define_sensor_transducer.
options_string = ' --p_raw';
if isfield(run_param, 'DEVICE_NUM')
    options_string = [options_string ' -g ' num2str(run_param.DEVICE_NUM)];
end

command = sprintf('%scd %s; ./%s -i %s -o %s%s', ...
    kwave_binary_env(run_param), ...
    strrep(binary_path, ' ', '\ '), binary_name, ...
    input_filename, output_filename, options_string);

status = system(command, '-echo');

% The input file is dead once the binary has read it, and it is gigabytes.
if exist(input_filename, 'file')
    delete(input_filename);
end

% A failure here is what killed the 2026-09-03 run, and it presented as
% "object 'Nx' doesn't exist" from h5read three frames of stack later. Say
% what actually happened, at the point it happened.
if status ~= 0
    if exist(output_filename, 'file')
        delete(output_filename);
    end
    error('run_simulation_to_disk:BinaryFailed', ...
        ['%s exited with status %d. A non-zero exit here is usually the ' ...
         'output file not fitting the disk; the preflight sizes it before ' ...
         'the run starts.'], binary_name, status);
end
if ~exist(output_filename, 'file')
    error('run_simulation_to_disk:NoOutput', ...
        '%s reported success but wrote no output file.', binary_name);
end

end
