function [response, eqparam, solverInfo] = calcBubbleResponse_GPU(liquid, ...
    gas, shell, bubble, pulse)
% GPU-accelerated Rayleigh-Plesset solver using fixed-step RK4.
% All microbubbles are solved in parallel on GPU with arrayfun kernel fusion.
% Drop-in replacement for calcBubbleResponse.m
%
% Optimizations:
%   - arrayfun kernel fusion (1 GPU kernel per RHS eval instead of ~20)
%   - configurable precision (single by default: 2x bandwidth on A100)
%   - strided integration (fewer loop iterations, interpolate output)
%
% Accuracy controls (all optional, read from the pulse struct):
%   pulse.rk4MaxPhaseStep  maximum phase advanced per RK4 substep [rad]
%   pulse.gpuPrecision     'single' (default) or 'double'
%   pulse.gpuMaxStride     upper bound on the output stride (1 disables it)
%   pulse.gpuPressureInterp  'pchip' (default) or 'linear'. The CPU solver
%                            evaluates a pchip interpolant of the transmit
%                            pressure at every stage time, so pchip is what
%                            reproduces it; 'linear' is kept for A/B runs.
%
% Nathan Blanken, University of Twente, 2023 (original CPU version)
% GPU adaptation, 2026

N_MB = length(bubble);
shell_model = validate_gpu_bubble_solver_inputs(pulse, shell);
precision   = resolve_gpu_precision(pulse);
strideLimit = resolve_gpu_max_stride(pulse);
pressureInterp = resolve_gpu_pressure_interp(pulse);
toPrecision = @(value) cast(value, precision);

%% Equation parameters (CPU — same as original, fast)
for i = N_MB:-1:1
    eqparam(i) = getEqParam(liquid, gas, shell(i), bubble(i), pulse);
end

%% Transfer per-bubble parameters to GPU as [1 x N_MB]
R0   = gpuArray(toPrecision([bubble.R0]));
kap  = gpuArray(toPrecision([eqparam.kappa]));
nu   = gpuArray(toPrecision([eqparam.nu]));
Ks   = gpuArray(toPrecision([shell.Ks]));
sig0 = gpuArray(toPrecision([shell.sig_0]));

P0  = toPrecision(liquid.P0);
rho = toPrecision(liquid.rho);
c_l = toPrecision(liquid.c);

%% Nondimensionalization (compute in double for accuracy, then cast)
T = toPrecision(median(sqrt(double(rho) * double(gather(R0)).^2 / double(P0))));
tq = pulse.tq;
N_out = length(tq);
dt_dim = toPrecision(tq(2) - tq(1));
dt = dt_dim / T;
fs_MB = toPrecision(1) / dt_dim;

%% Pressure on GPU as [N_MB x N_out]
P_gpu = gpuArray(toPrecision(pulse.p));

%% Surface tension model setup
switch shell_model
    case 'Marmottant'
        s_chi  = gpuArray(toPrecision([shell.chi]));
        s_Rb   = gpuArray(toPrecision([shell.Rb]));
        s_sigl = gpuArray(toPrecision([shell.sig_l]));
    case 'SegersTable'
        Am_tbl  = gpuArray(toPrecision(shell(1).sig.GridVectors{1}(:)'));
        sig_tbl = gpuArray(toPrecision(shell(1).sig.Values(:)'));
        s_AN    = gpuArray(toPrecision([shell.A_N]));
        s_Am1   = gpuArray(toPrecision([shell.A_m1]));
        s_Am2   = gpuArray(toPrecision([shell.A_m2]));
        s_sigl  = gpuArray(toPrecision([shell.sig_l]));
    case 'Segers'
        s_coeff = toPrecision(shell(1).coeff);
        s_AN    = gpuArray(toPrecision([shell.A_N]));
        s_Am1   = gpuArray(toPrecision([shell.A_m1]));
        s_Am2   = gpuArray(toPrecision([shell.A_m2]));
        s_sigl  = gpuArray(toPrecision([shell.sig_l]));
end

%% Fastest timescale that the integrator has to resolve
maxPhaseStep = resolve_gpu_rk4_max_phase_step(pulse);
[~, unstridedInfo] = calculate_gpu_rk4_substeps(eqparam, pulse, maxPhaseStep);
maxAngularFrequency = unstridedInfo.maxAngularFrequency;

%% Integration stride: reduce loop iterations by taking larger steps
% The pressure signal is band-limited by the transmit frequency, so the
% output grid may be coarser than the microbubble sampling rate. Substeps
% (below) keep the integration itself accurate at the strided step.
f_center = toPrecision(pulse.f);
max_harmonic = toPrecision(2);   % input pressure content up to 2nd harmonic
f_max = f_center * max_harmonic;
% Ensure at least 4 samples per period at highest input harmonic after striding
stride = max(1, floor(fs_MB / (toPrecision(4) * f_max)));
% Also limit by stability: strided step must not exceed the critical step
dt_crit_dim = toPrecision(2) / toPrecision(maxAngularFrequency);
stride = min(stride, max(1, floor(toPrecision(0.5) * dt_crit_dim / dt_dim)));
% Cap at the configured maximum (6 by default)
stride = double(min(stride, strideLimit));

dt_s = dt * toPrecision(stride);   % strided nondimensional step

%% Sub-steps within each strided step, bounded by the maximum phase step
[n_sub, solverInfo] = calculate_gpu_rk4_substeps(...
    eqparam, pulse, maxPhaseStep, stride);
solverInfo.stride = stride;
solverInfo.precision = precision;
solverInfo.pressureInterp = pressureInterp;
h = dt_s / toPrecision(n_sub);

%% Coarse output grid indices
idx_coarse = int32(1:stride:N_out);
if idx_coarse(end) ~= int32(N_out)
    idx_coarse(end+1) = int32(N_out);
end
N_coarse = length(idx_coarse);

fprintf(['    [GPU-RK4] N_MB=%d, N_out=%d, stride=%d, N_coarse=%d, ' ...
    'n_sub=%d, h=%.4g, precision=%s\n'], ...
    N_MB, N_out, stride, N_coarse, n_sub, h, precision);

%% Precompute RP equation constants as [1 x N_MB]
C1 = P0 * T^2 / rho ./ R0.^2;
C2 = gpuArray(toPrecision(1) + toPrecision(2)*sig0./(R0*P0));
C3 = gpuArray(toPrecision(3)*kap.*R0 / (c_l*T));
C4 = gpuArray(toPrecision(2)./(R0*P0));
C5 = gpuArray(toPrecision(4)*nu / (P0*T));
C6 = gpuArray(toPrecision(4)*Ks./(P0*R0*T));
invP0 = toPrecision(1) / P0;

%% Per-interval step sizes
% Appending the final sample to the coarse grid can leave a last interval
% that is shorter than the stride. Stepping the nominal strided h across it
% would integrate past the end of the pulse, so each interval gets the step
% its own width implies. The substep count stays as sized for the full
% stride, which only makes the short interval more accurate, not less.
h_interval = gpu_coarse_step_sizes(idx_coarse, dt, n_sub);
two = toPrecision(2);

%% Precompute pressure at coarse grid points to avoid per-iteration indexing
P_coarse = P_gpu(:, idx_coarse);  % [N_MB x N_coarse] — single bulk GPU op
dP_coarse = diff(P_coarse, 1, 2); % [N_MB x N_coarse-1]

%% Cubic Hermite (pchip) coefficients for the pressure between coarse samples
% Within a coarse interval the substeps need the pressure at intermediate
% times. Sampling it linearly is the dominant source of disagreement with the
% CPU reference, which evaluates a pchip interpolant instead. The shape-
% preserving slopes are cheap to precompute once per interval and turn the
% in-loop evaluation into three extra fused multiply-adds.
if strcmp(pressureInterp, 'pchip')
    t_coarse_dim = toPrecision(tq(idx_coarse));
    slopes = pchip_slopes(t_coarse_dim, P_coarse);   % [N_MB x N_coarse]
    % Scale by the interval width so the Hermite basis works on s in [0, 1].
    dt_coarse = gpuArray(toPrecision(diff(t_coarse_dim(:)')));  % [1 x N_coarse-1]
    Hm_lo = slopes(:, 1:end-1) .* dt_coarse;   % [N_MB x N_coarse-1]
    Hm_hi = slopes(:, 2:end)   .* dt_coarse;
else
    Hm_lo = [];
    Hm_hi = [];
end

%% Initialize state as [1 x N_MB]
x  = gpuArray(zeros(1, N_MB, precision));
xd = gpuArray(zeros(1, N_MB, precision));

%% Preallocate coarse output as [N_coarse x N_MB]
X_coarse  = gpuArray(zeros(N_coarse, N_MB, precision));
Xd_coarse = gpuArray(zeros(N_coarse, N_MB, precision));

% The fused kernels clamp 1+x away from zero to stay differentiable. Record
% where that clamp would have been hit so the run can be rejected instead of
% silently continuing from a collapsed bubble.
intermediateNonPositive = gpuArray(false(1, N_MB));

%% RK4 integration on coarse grid
% n_sub pre-computed interpolation fractions (avoid repeated division)
sub_fracs = toPrecision((0:n_sub) / n_sub);  % [0, 1/n_sub, ..., 1]

% Interpolation weights for the three stage times of every substep. Each row
% holds [start, midpoint, end]; the weights multiply the interval rise and,
% for pchip, the two scaled endpoint slopes.
stage_fracs = [sub_fracs(1:end-1)', ...
    (sub_fracs(1:end-1)' + sub_fracs(2:end)') * toPrecision(0.5), ...
    sub_fracs(2:end)'];                                  % [n_sub x 3]
[W_rise, W_lo, W_hi] = hermite_stage_weights(stage_fracs, pressureInterp);

for n = 1:(N_coarse-1)
    Pn = P_coarse(:, n)';    % [1 x N_MB] — precomputed, fast indexing
    dP = dP_coarse(:, n)';
    hn  = h_interval(n);
    hn2 = toPrecision(0.5) * hn;
    hn6 = hn / toPrecision(6);
    if isempty(Hm_lo)
        mLo = [];
        mHi = [];
    else
        mLo = Hm_lo(:, n)';
        mHi = Hm_hi(:, n)';
    end

    for s = 1:n_sub
        Ps = interp_pressure(1);
        Pm = interp_pressure(2);
        Pe = interp_pressure(3);

        track_invalid_state(x);
        [k1x, k1v] = rp_rhs(x,          xd,          Ps);
        k2StateX = x + hn2*k1x;
        track_invalid_state(k2StateX);
        [k2x, k2v] = rp_rhs(k2StateX,   xd+hn2*k1v,  Pm);
        k3StateX = x + hn2*k2x;
        track_invalid_state(k3StateX);
        [k3x, k3v] = rp_rhs(k3StateX,   xd+hn2*k2v,  Pm);
        k4StateX = x + hn*k3x;
        track_invalid_state(k4StateX);
        [k4x, k4v] = rp_rhs(k4StateX,   xd+hn*k3v,   Pe);

        x  = x  + hn6 * (k1x + two*k2x + two*k3x + k4x);
        xd = xd + hn6 * (k1v + two*k2v + two*k3v + k4v);
    end

    X_coarse(n+1,:)  = x;
    Xd_coarse(n+1,:) = xd;
end

%% Interpolate coarse results back to fine grid
t_coarse = gpuArray(toPrecision(tq(idx_coarse)));
t_fine   = gpuArray(toPrecision(tq));

if stride > 1
    % Gather to CPU for spline (avoids GPU NaN limitation), then send back
    tc = gather(t_coarse);
    tf = gather(t_fine);
    X_out  = interp1(tc, gather(X_coarse),  tf, 'spline');
    Xd_out = interp1(tc, gather(Xd_coarse), tf, 'spline');
else
    X_out  = X_coarse;
    Xd_out = Xd_coarse;
end

%% Gather and build response struct (convert back to double for compatibility)
X_out  = double(gather(X_out));
Xd_out = double(gather(Xd_out));
t_out  = tq(:);

validate_gpu_bubble_states(X_out, Xd_out, gather(intermediateNonPositive));

for i = N_MB:-1:1
    response(i).R    = bubble(i).R0 * (1 + X_out(:,i));
    response(i).Rdot = bubble(i).R0 * Xd_out(:,i) / double(T);
    response(i).t    = t_out;
end

%% ===== Nested helpers (share the integration workspace) =====
    function P = interp_pressure(stage)
        % Pressure at one RK4 stage time inside the current coarse interval.
        P = Pn + W_rise(s, stage) * dP;
        if ~isempty(mLo)
            P = P + W_lo(s, stage) * mLo + W_hi(s, stage) * mHi;
        end
    end

    function track_invalid_state(xi)
        intermediateNonPositive = intermediateNonPositive | 1 + xi <= 0;
    end

    function [dx, dv] = rp_rhs(xi, xdi, Pi)
        switch shell_model
            case 'Marmottant'
                % Fully fused: surface tension + RP in one kernel launch
                [dx, dv] = arrayfun(@rp_marmottant, xi, xdi, Pi, ...
                    R0, kap, C1, C2, C3, C4, C5, C6, invP0, ...
                    s_chi, s_Rb, s_sigl);

            otherwise
                % Compute surface tension separately, then fuse RP
                sig = compute_sig(xi);
                [dx, dv] = arrayfun(@rp_core, xi, xdi, Pi, sig, ...
                    kap, C1, C2, C3, C4, C5, C6, invP0);
        end
    end

    function sig = compute_sig(xi)
        Ri = R0 .* (1 + xi);
        switch shell_model
            case 'SegersTable'
                Am = toPrecision(4*pi)*Ri.^2 ./ s_AN;
                Am_c = min(max(Am, Am_tbl(1)), Am_tbl(end));
                sig = interp1(Am_tbl, sig_tbl, Am_c, 'linear');
                sig(Am <= s_Am1) = 0;
                sig(Am >= s_Am2) = s_sigl(Am >= s_Am2);
            case 'Segers'
                Am = toPrecision(4*pi)*Ri.^2 ./ s_AN;
                sig = polyval(s_coeff, Am);
                sig(Am < s_Am1) = 0;
                sig(Am > s_Am2) = s_sigl(Am > s_Am2);
        end
    end

end

%% ===== GPU arrayfun kernel functions =====
% These are local functions (not nested) so MATLAB can compile them for GPU.
% All literals stay untyped so the kernels run at the precision of their
% inputs.

function [dx, dv] = rp_marmottant(xi, xdi, Pi, ...
        R0i, kapi, c1, c2, c3, c4, c5, c6, invP0, ...
        chi, Rb, sigl)
    % Fused Marmottant surface tension + Rayleigh-Plesset RHS
    Ri = R0i * (1 + xi);
    sig_raw = chi * (Ri * Ri / (Rb * Rb) - 1);
    sig = min(max(sig_raw, 0), sigl);

    opx  = max(1 + xi, 1e-6);
    iopx = 1 / opx;
    dv = iopx * ( ...
        -1.5 * xdi * xdi ...
        + c1 * ( ...
            c2 * opx^(-3*kapi) * (1 - c3*xdi) ...
            - 1 ...
            - c4 * sig * iopx ...
            - c5 * xdi * iopx ...
            - c6 * xdi * iopx * iopx ...
            - Pi * invP0 ...
        ) ...
    );
    dx = xdi;
end

function [dx, dv] = rp_core(xi, xdi, Pi, sig, ...
        kapi, c1, c2, c3, c4, c5, c6, invP0)
    % Rayleigh-Plesset RHS with pre-computed surface tension
    opx  = max(1 + xi, 1e-6);
    iopx = 1 / opx;
    dv = iopx * ( ...
        -1.5 * xdi * xdi ...
        + c1 * ( ...
            c2 * opx^(-3*kapi) * (1 - c3*xdi) ...
            - 1 ...
            - c4 * sig * iopx ...
            - c5 * xdi * iopx ...
            - c6 * xdi * iopx * iopx ...
            - Pi * invP0 ...
        ) ...
    );
    dx = xdi;
end

function method = resolve_gpu_pressure_interp(pulse)
% Resolve how the transmit pressure is sampled between coarse grid points.
% The CPU reference uses a pchip interpolant, so that is the default here.

method = 'pchip';
if isfield(pulse, 'gpuPressureInterp')
    method = pulse.gpuPressureInterp;
end
if ~(ischar(method) || isstring(method)) || ...
        ~ismember(char(method), {'pchip', 'linear'})
    error('calcBubbleResponse_GPU:InvalidPressureInterp', ...
        'GPUPressureInterp must be ''pchip'' or ''linear''.');
end
method = char(method);
end

function precision = resolve_gpu_precision(pulse)
% Resolve the floating-point class used for the GPU integration.

precision = 'single';
if isfield(pulse, 'gpuPrecision')
    precision = pulse.gpuPrecision;
end
if ~(ischar(precision) || isstring(precision)) || ...
        ~ismember(char(precision), {'single', 'double'})
    error('calcBubbleResponse_GPU:InvalidPrecision', ...
        'GPUPrecision must be ''single'' or ''double''.');
end
precision = char(precision);
end

function maxStride = resolve_gpu_max_stride(pulse)
% Resolve the upper bound on the output stride. A value of 1 disables
% striding, so the integrator writes every microbubble sample.

maxStride = 6;
if isfield(pulse, 'gpuMaxStride')
    maxStride = pulse.gpuMaxStride;
end
if ~isscalar(maxStride) || ~isnumeric(maxStride) || ~isreal(maxStride) || ...
        ~isfinite(maxStride) || maxStride < 1 || ...
        maxStride ~= floor(maxStride)
    error('calcBubbleResponse_GPU:InvalidMaxStride', ...
        'GPUMaxStride must be a positive integer.');
end
maxStride = double(maxStride);
end
