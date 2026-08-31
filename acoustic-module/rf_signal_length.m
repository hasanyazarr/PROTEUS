function N = rf_signal_length(Transducer, Grid, M, dataType)
%RF_SIGNAL_LENGTH Padded receive record length used by compute_RF.
%
%   M samples plus the room the lens delays need, rounded up to a length
%   with small prime factors so the transforms are cheap. Its own function
%   because preflight_array_limits has to size compute_RF's arrays before
%   the run starts, and a second copy of this arithmetic would drift.

delays = cast(Transducer.integration_receive_delays(:), dataType);

% Compute signal length required to apply the delays:
N = M + ceil(max(delays)/Grid.dt);

% Get a signal length with small prime factors:
max_expansion = max(10,round(N*0.05));
max_prime = 5;
N = optimize_grid_size(N, [0 max_expansion], max_prime);

end
