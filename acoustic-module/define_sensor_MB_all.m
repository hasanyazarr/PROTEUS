function [sensor, MB_idx_all, max_mb, bubble_counts] = define_sensor_MB_all(...
    Grid, folder, Acquisition, N_sequence, Geometry)
%DEFINE_SENSOR_MB_ALL loops through all the frames and sequence pulses that
%need to be simulated and adds all corresponding microbubbles to the sensor
%struct for the first iteration.
%
% Grid:        Grid properties
% folder:      folder containing ground truth microbubble positions
% Acquisition: Acquisition properties
% N_sequence:  Number of pulses in each acquisition sequence
% Geometry:    Geometry properties
%
% Nathan Blanken, Alina Kuliesh, Guillaume Lajoinie, 2023

% Build the sensor mask over the UNION of bubble positions in the current
% Acquisition.StartFrame..Acquisition.EndFrame window. The total
% NumberOfFrames is still used for ground-truth filename padding.
Nframes     = Acquisition.NumberOfFrames; % Total number of ground truth frames
frame_start = Acquisition.StartFrame;
frame_end   = Acquisition.EndFrame;

% Logical, not double: at v11's lambda/8 grid the mask is ~300M points,
% which is 2.4 GB as double against 0.3 GB as logical, and it is
% allocated and zeroed once per frame per pulse. define_sensor_transducer
% has always built its mask this way.
sensor.mask = zeros(Grid.Nx, Grid.Ny, Grid.Nz, 'logical');
max_mb = 1;
bubble_counts = zeros(frame_end - frame_start + 1, N_sequence);

for frame = frame_start : frame_end
       
    for pulse_seq_idx = 1:N_sequence
    
        MB = load_microbubbles(folder, frame, pulse_seq_idx, Geometry, Nframes);

        % Put the microbubbles on the grid:
        [MB.points, ~, MB_idx, ~] = voxelize_media_points(MB.points, Grid);
        frameIndex = frame - frame_start + 1;
        bubble_counts(frameIndex, pulse_seq_idx) = size(MB.points, 1);
        
        if size(MB.points, 1) > max_mb
            max_mb = size(MB.points, 1);
        end

        % Put sensor at the microbubbles
        mask_only = true;
        [sensor,~] = update_sensor(sensor, MB.points, MB_idx, ...
            Grid, mask_only);
    
    end

    
end

sensor.record={'p'};
MB_idx_all = find(sensor.mask == 1);

end
