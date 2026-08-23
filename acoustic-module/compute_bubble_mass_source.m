function [mass_source, runInfo] = compute_bubble_mass_source(...
    sensed_p,  radii, kgrid, Medium, Microbubble, Transmit)
%COMPUTE_BUBBLE_MASS_SOURCE computes the response of microbubbles to
%pressure signals sensed_p and converts the response to a mass source
%matrix.
%
% Nathan Blanken, Guillaume Lajoinie, University of Twente, 2023

% Number of microbubbles and signal length:
[N_MB,N] = size(sensed_p);

% GPU integration is opt-in because it uses a different numerical solver.
useGPU = false;
if isfield(Microbubble, 'UseGPU')
    if ~isscalar(Microbubble.UseGPU)
        error('compute_bubble_mass_source:InvalidUseGPU', ...
            'Microbubble.UseGPU must be a scalar logical value.');
    end
    useGPU = logical(Microbubble.UseGPU);
end
gpuPrecision = resolve_gpu_precision_setting(Microbubble);

% Sampling rate for the microbubble module:
fs_MB = Microbubble.SamplingRate;

% Time vectors for k-Wave signals and microbubble module signals:
t_kwave = (0:(N-1))*kgrid.dt;
M = floor(t_kwave(end)*fs_MB) + 1;
t_MB    = (0:(M-1)) / fs_MB;

if N_MB == 0
    mass_source = zeros(0, N, class(sensed_p));
    runInfo = empty_run_info(useGPU, M, class(sensed_p));
    return
end

if useGPU
    hasPCT = license('test', 'Distrib_Computing_Toolbox');
    if ~hasPCT || gpuDeviceCount("available") == 0
        error('compute_bubble_mass_source:GPUUnavailable', ...
            ['Microbubble.UseGPU is enabled, but Parallel Computing ' ...
             'Toolbox and an available GPU are required.']);
    end
end

% Divide the microbubbles into batches. Older settings without GPU-specific
% fields retain their configured CPU batch size.
if useGPU
    batchSize = select_gpu_batch_size(N_MB, M, Microbubble, gpuPrecision);
else
    batchSize = Microbubble.BatchSize;
end
Nbatch = ceil(N_MB/batchSize); % Total number of batches

runInfo = empty_run_info(useGPU, M, class(sensed_p));
runInfo.batchSize = batchSize;
runInfo.numberOfBatches = Nbatch;
runInfo.numberOfInputBubbles = N_MB;

% Decide whether to use parfor loop or not:
if useGPU
    % Multiple workers sharing one GPU add contention rather than throughput.
    useparfor = false;
else
    switch Microbubble.UseParfor
        case 'on'
            useparfor = true;
        case 'off'
            useparfor = false;
        case 'auto'
            if Nbatch>8
                useparfor = true;
            else
                useparfor = false;
            end
    end
end

% Resample signal at the sampling rate of the microbubble module:
sensed_p = sinc_interpolation(t_kwave, transpose(sensed_p), t_MB);

% Filter settings:
Filter.dt          = 1/fs_MB;
Filter.t_array     = [];
Filter.k_max       = kgrid.k_max;
Filter.TW          = 0.1/(fs_MB*kgrid.dt);
Filter.sound_speed = Medium.SpeedOfSoundMinimum;

% Microbubble driving pulse settings:
pulse.f  = Transmit.CenterFrequency;
pulse.w  = pulse.f * 2 * pi;    
pulse.fs = fs_MB;
pulse.tq = t_MB;
pulse.t  = t_MB;
pulse.dispProgress = false; % Do not show ODE solver progress
pulse.gpuPrecision = gpuPrecision;
pulse.rk4MaxPhaseStep = resolve_gpu_rk4_max_phase_step(Microbubble);
if isfield(Microbubble, 'GPUMaxStride')
    pulse.gpuMaxStride = Microbubble.GPUMaxStride;
end
if isfield(Microbubble, 'GPUPressureInterp')
    pulse.gpuPressureInterp = Microbubble.GPUPressureInterp;
end

% Get the properties of the liquid, gas, and shell:
[liquid, gas] = get_microbubble_material_properties(Medium, Microbubble);

% Convert radii into a row vector to ensure bubble and shell are also row
% vectors:
radii = transpose(radii);

% Preallocate the mass source matrix:
mass_source = zeros(N_MB,M,class(sensed_p));

% Loop over all MBs to compute response:
t_ode = tic;
if useparfor
    
    % Write the sensed pressure and the radii to cell arrays, each cell
    % holding the values for one batch:
    sensed_p_cell = cell(1,Nbatch);
	radii_cell = cell(1,Nbatch);
    
    for k = 1:Nbatch
        
        % Microbubble indices in the current batch:
        idx = get_batch_indices(k, N_MB, batchSize, radii);
        
        sensed_p_cell{k} = sensed_p(idx,:);
        radii_cell{k} = radii(idx);
    end
    
    % Cell array for holding the results of the parfor loop:
    mass_source_cell = cell(1,Nbatch);
    
    parfor k = 1:Nbatch

        mass_source_cell{k} = compute_mass_source(sensed_p_cell{k}, ...
            radii_cell{k}, Medium, Microbubble, liquid, gas, pulse, useGPU);

    end
    
    % Write the results in the the cell array to a matrix:
    for k = 1:Nbatch
        
        % Microbubble indices in the current batch:
        idx = get_batch_indices(k, N_MB, batchSize, radii);
        
        mass_source(idx,:) = mass_source_cell{k};
    end
    
else
    batchSolverInfo = cell(1, Nbatch);
    for k = 1:Nbatch

        % Microbubble indices in the current batch:
        idx = get_batch_indices(k, N_MB, batchSize, radii);

        try
            [mass_source(idx,:), batchSolverInfo{k}] = compute_mass_source(...
                sensed_p(idx,:), radii(idx), Medium, Microbubble, ...
                liquid, gas, pulse, useGPU);
        catch exception
            if useGPU
                context = MException(...
                    'compute_bubble_mass_source:GPUBatchFailed', ...
                    ['GPU bubble batch %d/%d failed for original ' ...
                     'bubble indices [%s].'], ...
                    k, Nbatch, strjoin(string(idx), ','));
                context = addCause(context, exception);
                throw(context)
            end
            rethrow(exception)
        end

    end

    if useGPU
        runInfo.rk4SubstepsPerBatch = cellfun(...
            @(info) info.substeps, batchSolverInfo);
        runInfo.rk4MaxAngularFrequencyPerBatch = cellfun(...
            @(info) info.maxAngularFrequency, batchSolverInfo);
        runInfo.rk4ActualPhaseStepPerBatch = cellfun(...
            @(info) info.actualPhaseStep, batchSolverInfo);
        runInfo.stridePerBatch = cellfun(...
            @(info) info.stride, batchSolverInfo);
        runInfo.rk4MaxPhaseStep = batchSolverInfo{1}.maxPhaseStep;
        runInfo.gpuPrecision = batchSolverInfo{1}.precision;
        runInfo.gpuPressureInterp = batchSolverInfo{1}.pressureInterp;
    end
end

% One provenance banner per run instead of a solver line per batch per frame.
if useGPU
    run_log('banner', 'solver', ...
        ['MB solver: GPU-RK4 | N_MB=%d (%d batch x %d) | N_out=%d\n' ...
         '    stride=%d, n_sub=%d, precision=%s, pressure=%s'], ...
        N_MB, Nbatch, batchSize, M, batchSolverInfo{1}.stride, ...
        batchSolverInfo{1}.substeps, batchSolverInfo{1}.precision, ...
        batchSolverInfo{1}.pressureInterp);
else
    run_log('banner', 'solver', ...
        'MB solver: CPU ode45 | N_MB=%d (%d batch x %d) | N_out=%d', ...
        N_MB, Nbatch, batchSize, M);
end
run_log('stage', 'ODE', toc(t_ode));

% Filter unsupported frequencies before downsampling:
% Batch FFT-domain low-pass filter (replaces per-bubble filterTimeSeries loop)
f_max_filter = Filter.k_max * Filter.sound_speed / (2 * pi);
filter_cutoff_f = f_max_filter;  % PPW=2 → cutoff = f_max
tw_hz = Filter.TW * fs_MB;      % transition width in Hz

% Build frequency axis for the batch
N_filt = size(mass_source, 2);
f_hz = (0:N_filt-1) * (fs_MB / N_filt);
f_hz(ceil(N_filt/2+1):end) = f_hz(ceil(N_filt/2+1):end) - fs_MB;

% Zero-phase low-pass: smooth rolloff matching Kaiser window behavior
f_norm = max(0, (abs(f_hz) - filter_cutoff_f) / tw_hz);
H = 0.5 * (1 + cos(pi * min(f_norm, 1)));
H(f_norm > 1) = 0;

% Apply to all signals at once via FFT (zero-phase = magnitude-only in freq domain)
MS_fft = fft(mass_source, [], 2);
mass_source = real(ifft(MS_fft .* H, [], 2));

% Resample signals at the sampling rate of the acoustic module:
mass_source = sinc_interpolation(t_MB, transpose(mass_source), t_kwave);


end


%==========================================================================
% FUNCTIONS
%==========================================================================

function runInfo = empty_run_info(useGPU, numberOfOutputSamples, inputDtype)
% Diagnostics reported back to the caller. The GPU fields stay empty on the
% CPU path.

runInfo.useGPU = useGPU;
runInfo.batchSize = 1;
runInfo.numberOfBatches = 0;
runInfo.numberOfInputBubbles = 0;
runInfo.numberOfOutputSamples = numberOfOutputSamples;
runInfo.inputDtype = inputDtype;
runInfo.rk4SubstepsPerBatch = [];
runInfo.rk4MaxAngularFrequencyPerBatch = [];
runInfo.rk4ActualPhaseStepPerBatch = [];
runInfo.stridePerBatch = [];
runInfo.rk4MaxPhaseStep = NaN;
runInfo.gpuPrecision = '';
runInfo.gpuPressureInterp = '';
end

function precision = resolve_gpu_precision_setting(Microbubble)
% Resolve the GPU floating-point class. Older settings default to the
% single-precision solver the optimized GPU path was tuned for.

precision = 'single';
if isfield(Microbubble, 'GPUPrecision')
    precision = Microbubble.GPUPrecision;
end
if ~(ischar(precision) || isstring(precision)) || ...
        ~ismember(char(precision), {'single', 'double'})
    error('compute_bubble_mass_source:InvalidGPUPrecision', ...
        'Microbubble.GPUPrecision must be ''single'' or ''double''.');
end
precision = char(precision);
end

function batchSize = select_gpu_batch_size(N_MB, N_out, Microbubble, precision)
% Select a GPU batch size from current free memory or a manual override.

gpuBatchSetting = Microbubble.BatchSize;
if isfield(Microbubble, 'GPUBatchSize')
    gpuBatchSetting = Microbubble.GPUBatchSize;
end

if ischar(gpuBatchSetting) || ...
        (isstring(gpuBatchSetting) && isscalar(gpuBatchSetting))
    if ~strcmpi(gpuBatchSetting, 'auto')
        error('compute_bubble_mass_source:InvalidGPUBatchSize', ...
            'Microbubble.GPUBatchSize must be ''auto'' or a positive integer.');
    end

    memoryFraction = 0.50;
    if isfield(Microbubble, 'GPUMemoryFraction')
        memoryFraction = Microbubble.GPUMemoryFraction;
    end
    if ~isnumeric(memoryFraction) || ~isscalar(memoryFraction) || ...
            ~isfinite(memoryFraction) || memoryFraction <= 0 || ...
            memoryFraction > 1
        error('compute_bubble_mass_source:InvalidGPUMemoryFraction', ...
            'Microbubble.GPUMemoryFraction must be in the interval (0, 1].');
    end

    maxBatchSize = inf;
    if isfield(Microbubble, 'GPUMaxBatchSize')
        maxBatchSize = Microbubble.GPUMaxBatchSize;
    end
    if ~isnumeric(maxBatchSize) || ~isscalar(maxBatchSize) || ...
            isnan(maxBatchSize) || maxBatchSize <= 0 || ...
            (isfinite(maxBatchSize) && maxBatchSize ~= floor(maxBatchSize))
        error('compute_bubble_mass_source:InvalidGPUMaxBatchSize', ...
            'Microbubble.GPUMaxBatchSize must be a positive integer or inf.');
    end

    device = gpuDevice;
    availableBytes = double(device.AvailableMemory);

    % Pressure, radius, and velocity histories dominate persistent GPU
    % storage. The factor of two reserves space for RK4 temporaries and
    % allocator overhead.
    bytesPerSample = 8;
    if strcmp(precision, 'single')
        bytesPerSample = 4;
    end
    bytesPerBubble = 2 * 3 * N_out * bytesPerSample;
    automaticBatchSize = floor(...
        availableBytes * memoryFraction / bytesPerBubble);
    if automaticBatchSize < 1
        error('compute_bubble_mass_source:InsufficientGPUMemory', ...
            ['Available GPU memory is below the configured safe budget ' ...
             'for one microbubble.']);
    end

    batchSize = max(1, min([N_MB, maxBatchSize, automaticBatchSize]));
else
    if ~isnumeric(gpuBatchSetting) || ~isscalar(gpuBatchSetting) || ...
            ~isfinite(gpuBatchSetting) || gpuBatchSetting <= 0 || ...
            gpuBatchSetting ~= floor(gpuBatchSetting)
        error('compute_bubble_mass_source:InvalidGPUBatchSize', ...
            'Microbubble.GPUBatchSize must be ''auto'' or a positive integer.');
    end
    batchSize = gpuBatchSetting;
end

end

function idx = get_batch_indices(batchIndex, N_MB, batchSize, radii)
% Sort the bubbles by size and group them into batches of batchSize.
% Bubbles of similar size have similar characteristic timescales. Having
% similarly sized microbubbles in a batch is expected to speed up
% computation.

% Linearly increasing indices:
idxLinear = (batchIndex-1)*batchSize + (1:batchSize);
idxLinear(idxLinear>N_MB) = [];

% Microbubble indices sorted by size:
[~,idxSort] = sort(radii);
idx = idxSort(idxLinear);
end

function [mass_source, solverInfo] = compute_mass_source(...
    sensed_p,  radii, Medium, Microbubble, liquid, gas, pulse, useGPU)

% Microbubble driving pulses:
pulse.p = sensed_p;

% Shell properties:
shell = arrayfun(@(x) ...
    get_microbubble_shell_properties(x,Medium,Microbubble),radii);

% Bubble properties (radius in meters):
bubble = arrayfun(@(x) struct('R0',x), radii);

% Compute the bubble response:
if useGPU
    [response, ~, solverInfo] = calcBubbleResponse_GPU(...
        liquid, gas, shell, bubble, pulse);
else
    response = calcBubbleResponse(liquid, gas, shell, bubble, pulse);
    solverInfo = struct();
end

% Compute mass source for the current batch:
R    = transpose([response.R]);
Rdot = transpose([response.Rdot]);   
mass_source = 4*pi*liquid.rho*R.^2 .*Rdot;

end

function [liquid, gas] = get_microbubble_material_properties(...
    Medium, Microbubble)
% Convert the material properties from the GUI to the format used by the
% microbubble module.

Liquid     = Medium.Vessel;
Gas        = Microbubble.Gas;

% Properties of the liquid:
liquid.k   = Liquid.ThermalConductivity;    % [W/m/K]
liquid.rho = Liquid.Density;                % [kg/m^3]
liquid.cp  = Liquid.SpecificHeat;   	    % [J/kg/K]
liquid.nu  = Liquid.DynamicViscosity;       % [Pa.s]
liquid.c   = Liquid.SpeedOfSound;           % [m/s]

% Environmental conditions:
liquid.T0  = Liquid.Temperature;            % [K]
liquid.P0  = Liquid.Pressure;               % [Pa]

% Thermodynanic model for microbubble oscillations ('Adiabatic', 
% 'Isothermal', or 'Propsperetti'):
liquid.ThermalModel = Microbubble.ThermalModel;

% Properties of the gas:
gas.k      = Gas.ThermalConductivity;     	% [W/m/K]
gas.rho    = Gas.Density;                   % [kg/m^3]
gas.Mg     = Gas.MolarMass;                 % [kg/mol]
gas.gam    = Gas.HeatCapacityRatio;
gas.cp     = Gas.SpecificHeat;              % Constant pressure [J/kg/K]

end


function shell = get_microbubble_shell_properties(R0, Medium, Microbubble)
% Shell properties of a microbubble. According to:
% Marmottant et al., J. Acoust. Soc. Am. 118 6, 2005
% OR
% Segers et al., Soft Matter, 2018, 14, 9550-9561

Shell  = Microbubble.Shell;
Liquid = Medium.Vessel;

shell.sig_0 = Shell.InitialSurfaceTension;

if strcmp(Shell.Model, 'Segers') || strcmp(Shell.Model, 'Custom')
    shell.model = 'SegersTable';
else
    shell.model = Shell.Model;
end


% MODEL CHECK AND MAXIMUM SURFACE TENSION

if R0<0.5e-6 || R0>6e-6
    warning('Microbubble dynamics uncertain for given microbubble radius.')
end

% SURFACE TENSION CURVES

switch shell.model
    case 'Marmottant'
        % MARMOTTANT MODEL    
        
        % Linearised surface tension curve (Marmottant et al., J. Acoust. 
        % Soc. Am. 118 6, December 2005)

        shell.chi   = Shell.Elasticity; % [N/m]

        % Compute the buckling radius (m):
        shell.Rb    = R0/sqrt(1+shell.sig_0/shell.chi);

        % Maximum surface tension [N/m]
        shell.sig_l = Liquid.SurfaceTension;
    
    case {'SegersTable','Custom'}
        % EXPERIMENTAL SURFACE TENSION CURVES (TABLE LOOKUP)

        A_0 = 4*pi*R0^2;         % Initial microbubble area

        % Experimental surface tension curve:
        fit.sig      = Microbubble.Shell.SurfaceTension;            
        fit.A_m_list = Microbubble.Shell.NormalizedArea;

        shell.A_m1 = min(fit.A_m_list);  % Left  domain boundary fit.
        shell.A_m2 = max(fit.A_m_list);  % Right domain boundary fit.

        % Find the normalized area for which the fit equals sig_0:
        A_m0 = interp1(fit.sig, fit.A_m_list, shell.sig_0);

        shell.A_N  = A_0/A_m0; % Reference area surface tension curve.
        
        shell.sig = griddedInterpolant(fit.A_m_list,fit.sig);

        % Surface tension of the surrounding liquid:
        shell.sig_l = max(fit.sig);
    
    otherwise
        error('Unknown shell model')
    
end

% SHELL VISCOSITY
if Microbubble.Advanced
    shell.Ks = Shell.Viscosity; % [N.s/m]    
else
    % Shell viscosity, Segers et al, Soft Matter, 14, 2018
    % Surface dilatational viscosity (N.s/m). Fit to figure 6B:
    c_1=1.5e-9; 
    c_2=8e5; 
    shell.Ks = c_1.*exp(c_2.*R0); 
end


end