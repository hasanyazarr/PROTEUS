function sensor_data = full_simulator(...
    source, ...
    sensor_transducer,...
    sensor_frame, sensor_weights_frame, sensor_mask_idx_frame,...
    sensed_p,...
    MB, kgrid, Grid, medium, run_param, ...
    Medium, Microbubble, Transmit)

t_end_3         = run_param.tr(3);
max_trans_dist  = run_param.max_trans_dist;
max_dist        = run_param.max_dist;
pulse_length    = run_param.pulse_length;
N_interactions  = run_param.N_interactions;

% Grid size for single-bubble simulations:
% (min 2 lambda between source and PML)
f0 = Transmit.CenterFrequency;
N_sup = ceil(2 * 2 * Medium.SpeedOfSound / f0 / Grid.dx); % [voxels]

for iter = 1:N_interactions

    % create the time array
    t_end_2 = (max_trans_dist + max_dist  + 2*pulse_length) / ...
        Medium.SpeedOfSound; % [s]

    % Update t_array (array updates automatically):
    kgrid.Nt = floor(t_end_2 / kgrid.dt) + 1; 

    % Compute microbubble mass sources:       
    t_mb = tic;
    mass_source = compute_bubble_mass_source(...
        sensed_p,  MB.radii, kgrid, Medium, Microbubble, Transmit);
    run_log('stage', 'MB', toc(t_mb));

    % Add the microbubble mass sources to the source:           
    source = update_source(source, mass_source, ...
        transpose(sensor_weights_frame), sensor_mask_idx_frame, ...
        Grid, medium);

    % Run the k-Wave simulation:
    t_prop = tic;
    sensor_data = run_simulation(...
        run_param, kgrid, medium, source, sensor_frame);
    run_log('stage', 'prop', toc(t_prop));

    % Pressure sensed by the microbubbles:     
    sensed_p = sensor_weights_frame*double(sensor_data.p);
    sensed_p = cast(full(sensed_p), class(sensor_data.p));

    % Subtract the self-sensed pressure:
    self_sense_pressure = compute_self_sense_pressure(kgrid, ...
        Grid, MB.idx, MB.points, mass_source,...
        medium, N_sup, run_param);

    sensed_p = sensed_p - self_sense_pressure;

end

% Third iteration: transducer send & record pulse ; MBs send pulse

% Update t_array (array updates automatically):
kgrid.Nt = floor(t_end_3 / kgrid.dt) + 1;

% Compute microbubble mass sources:       
t_mb = tic;
mass_source = compute_bubble_mass_source(...
    sensed_p,  MB.radii, kgrid, Medium, Microbubble, Transmit);
run_log('stage', 'MB', toc(t_mb));

% Add the microbubble mass sources to the source:       
source = update_source(source, mass_source, ...
    transpose(sensor_weights_frame), sensor_mask_idx_frame, Grid, medium);

t_prop = tic;
sensor_data = run_simulation(...
    run_param, kgrid, medium, source, sensor_transducer);
run_log('stage', 'prop', toc(t_prop));

end