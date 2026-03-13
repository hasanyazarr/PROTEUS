function [response, eqparam] = calcBubbleResponse_GPU(liquid, gas, ...
    shell, bubble, pulse)
% GPU-accelerated Rayleigh-Plesset solver using fixed-step RK4.
% All microbubbles are solved in parallel on GPU with arrayfun kernel fusion
% and single-precision arithmetic for maximum throughput.
% Drop-in replacement for calcBubbleResponse.m
%
% Nathan Blanken, University of Twente, 2023 (original CPU version)
% GPU adaptation, 2026

N_MB = length(bubble);

%% Equation parameters (CPU — same as original, fast)
for i = N_MB:-1:1
    eqparam(i) = getEqParam(liquid, gas, shell(i), bubble(i), pulse);
end

%% Transfer per-bubble parameters to GPU as single [1 x N_MB]
R0   = gpuArray(single([bubble.R0]));
kap  = gpuArray(single([eqparam.kappa]));
nu   = gpuArray(single([eqparam.nu]));
Ks   = gpuArray(single([shell.Ks]));
sig0 = gpuArray(single([shell.sig_0]));

P0  = single(liquid.P0);
rho = single(liquid.rho);
c_l = single(liquid.c);

%% Nondimensionalization (compute in double for accuracy, then cast)
T = single(median(sqrt(double(rho) * double(gather(R0)).^2 / double(P0))));
tq = pulse.tq;
N_out = length(tq);
dt_dim = single(tq(2) - tq(1));
dt = dt_dim / T;

%% Pressure on GPU as single [N_MB x N_out]
P_gpu = gpuArray(single(pulse.p));

%% Surface tension model setup
shell_model = shell(1).model;
switch shell_model
    case 'Marmottant'
        s_chi  = gpuArray(single([shell.chi]));
        s_Rb   = gpuArray(single([shell.Rb]));
        s_sigl = gpuArray(single(shell(1).sig_l));
    case 'SegersTable'
        Am_tbl  = gpuArray(single(shell(1).sig.GridVectors{1}(:)'));
        sig_tbl = gpuArray(single(shell(1).sig.Values(:)'));
        s_AN    = gpuArray(single([shell.A_N]));
        s_Am1   = gpuArray(single([shell.A_m1]));
        s_Am2   = gpuArray(single([shell.A_m2]));
        s_sigl  = gpuArray(single([shell.sig_l]));
    case 'Segers'
        s_coeff = single(shell(1).coeff);
        s_AN    = gpuArray(single([shell.A_N]));
        s_Am1   = gpuArray(single([shell.A_m1]));
        s_Am2   = gpuArray(single([shell.A_m2]));
        s_sigl  = gpuArray(single([shell.sig_l]));
end

%% Stability: sub-steps per output step (includes C1 for safety)
C1_pre = P0 * T^2 / rho ./ R0.^2;
omega_nd = sqrt(single(3) * max(gather(kap .* C1_pre)));
dt_crit  = single(2.0) / omega_nd;
n_sub    = max(1, ceil(dt / dt_crit));
h        = dt / single(n_sub);

fprintf('    [GPU-RK4] N_MB=%d, N_out=%d, n_sub=%d, dt=%.4g, h=%.4g\n', ...
    N_MB, N_out, n_sub, dt, h);

%% Precompute RP equation constants as single [1 x N_MB]
C1 = C1_pre;
C2 = gpuArray(single(1) + single(2)*sig0./(R0*P0));
C3 = gpuArray(single(3)*kap.*R0 / (c_l*T));
C4 = gpuArray(single(2)./(R0*P0));
C5 = gpuArray(single(4)*nu / (P0*T));
C6 = gpuArray(single(4)*Ks./(P0*R0*T));
invP0 = single(1) / P0;

%% Precompute step-size fractions
h2 = single(0.5) * h;
h6 = h / single(6);
two = single(2);

%% Initialize state as single [1 x N_MB]
x  = gpuArray(zeros(1, N_MB, 'single'));
xd = gpuArray(zeros(1, N_MB, 'single'));

%% Preallocate output as single [N_out x N_MB]
X_out  = gpuArray(zeros(N_out, N_MB, 'single'));
Xd_out = gpuArray(zeros(N_out, N_MB, 'single'));

%% RK4 integration
for n = 1:(N_out-1)
    Pn  = P_gpu(:, n)';      % [1 x N_MB]
    Pn1 = P_gpu(:, n+1)';
    dP  = Pn1 - Pn;

    for s = 0:(n_sub-1)
        a0 = single(s)       / single(n_sub);
        ah = (single(s) + single(0.5)) / single(n_sub);
        a1 = single(s + 1)   / single(n_sub);

        Ps = Pn + a0 * dP;
        Pm = Pn + ah * dP;
        Pe = Pn + a1 * dP;

        [k1x, k1v] = rp_rhs(x,          xd,          Ps);
        [k2x, k2v] = rp_rhs(x+h2*k1x,   xd+h2*k1v,   Pm);
        [k3x, k3v] = rp_rhs(x+h2*k2x,   xd+h2*k2v,   Pm);
        [k4x, k4v] = rp_rhs(x+h*k3x,    xd+h*k3v,    Pe);

        x  = x  + h6 * (k1x + two*k2x + two*k3x + k4x);
        xd = xd + h6 * (k1v + two*k2v + two*k3v + k4v);
    end

    X_out(n+1,:)  = x;
    Xd_out(n+1,:) = xd;
end

%% Gather and build response struct (convert back to double for compatibility)
X_out  = double(gather(X_out));
Xd_out = double(gather(Xd_out));
t_out  = tq(:);

for i = N_MB:-1:1
    response(i).R    = bubble(i).R0 * (1 + X_out(:,i));
    response(i).Rdot = bubble(i).R0 * Xd_out(:,i) / double(T);
    response(i).t    = t_out;
end

%% ===== Nested RHS: arrayfun-fused kernels =====
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
        Ri = R0 .* (single(1) + xi);
        switch shell_model
            case 'SegersTable'
                Am = single(4*pi)*Ri.^2 ./ s_AN;
                Am_c = min(max(Am, Am_tbl(1)), Am_tbl(end));
                sig = interp1(Am_tbl, sig_tbl, Am_c, 'linear');
                sig(Am <= s_Am1) = single(0);
                sig(Am >= s_Am2) = s_sigl(Am >= s_Am2);
            case 'Segers'
                Am = single(4*pi)*Ri.^2 ./ s_AN;
                sig = polyval(s_coeff, Am);
                sig(Am < s_Am1) = single(0);
                sig(Am > s_Am2) = s_sigl(Am > s_Am2);
        end
    end

end

%% ===== GPU arrayfun kernel functions =====
% These are local functions (not nested) so MATLAB can compile them for GPU.

function [dx, dv] = rp_marmottant(xi, xdi, Pi, ...
        R0i, kapi, c1, c2, c3, c4, c5, c6, invP0, ...
        chi, Rb, sigl)
    % Fused Marmottant surface tension + Rayleigh-Plesset RHS
    Ri = R0i * (1 + xi);
    sig = chi * (Ri * Ri / (Rb * Rb) - 1);
    if sig < 0;    sig = 0;    end
    if sig > sigl; sig = sigl; end

    opx  = 1 + xi;
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
    opx  = 1 + xi;
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
