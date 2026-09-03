function medium_average = average_medium(medium)
%AVERAGE_MEDIUM Homogeneous-medium approximation of a k-Wave medium struct.
%
%   Reduces the full property grids to the four scalars the homogeneous
%   propagation needs: density, sound speed, absorption prefactor and
%   absorption power.
%
%   This used to be a subfunction of run_simulation_homogeneous, called on
%   every entry. Each call reads sound_speed, density and alpha_coeff in
%   full: at v11's lambda/8 grid that is ~300M points x 8 bytes x 3 grids =
%   7.2 GB of host memory traffic, and the frame loop calls it once per
%   pulse per frame. The medium is built once in main_RF and never changes
%   after, so main_RF computes this once and hands it down in
%   run_param.MediumAverage; the call here stays as the fallback for direct
%   callers and for the MATLAB tests, which build run_param themselves.

medium_average.rho = mean(medium.density,     'all');
medium_average.c   = mean(medium.sound_speed, 'all');
medium_average.a   = mean(medium.alpha_coeff, 'all');
medium_average.b   = mean(medium.alpha_power, 'all');

if isfield(medium,'alpha_mode')
    medium_average.alpha_mode = medium.alpha_mode;
end

end
