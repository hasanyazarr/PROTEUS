function [response, eqparam] = calcBubbleResponse_GPU(liquid, gas, ...
    shell, bubble, pulse)
% GPU-accelerated Rayleigh-Plesset solver using fixed-step RK4.
% All microbubbles are solved in parallel on GPU.
% Drop-in replacement for calcBubbleResponse.m
%
% Nathan Blanken, University of Twente, 2023 (original CPU version)
% GPU adaptation, 2026

N_MB = length(bubble);

%% Equation parameters (CPU — same as original, fast)
for i = N_MB:-1:1
    eqparam(i) = getEqParam(liquid, gas, shell(i), bubble(i), pulse);
end

%% Transfer per-bubble parameters to GPU [1 x N_MB]
R0   = gpuArray([bubble.R0]);
kap  = gpuArray([eqparam.kappa]);
nu   = gpuArray([eqparam.nu]);
Ks   = gpuArray([shell.Ks]);
sig0 = gpuArray([shell.sig_0]);

P0  = liquid.P0;
rho = liquid.rho;
c_l = liquid.c;

%% Nondimensionalization (same as original)
T = median(sqrt(rho * gather(R0).^2 / P0));
tq = pulse.tq;
N_out = length(tq);
dt_dim = tq(2) - tq(1);   % uniform dimensional time step
dt = dt_dim / T;           % nondimensional step

%% Pressure on GPU [N_MB x N_out]
P_gpu = gpuArray(pulse.p);

%% Surface tension model setup
shell_model = shell(1).model;
switch shell_model
    case 'Marmottant'
        s_chi  = gpuArray([shell.chi]);
        s_Rb   = gpuArray([shell.Rb]);
        s_sigl = gpuArray(shell(1).sig_l);
    case 'SegersTable'
        Am_tbl  = gpuArray(shell(1).sig.GridVectors{1}(:)');
        sig_tbl = gpuArray(shell(1).sig.Values(:)');
        s_AN    = gpuArray([shell.A_N]);
        s_Am1   = gpuArray([shell.A_m1]);
        s_Am2   = gpuArray([shell.A_m2]);
        s_sigl  = gpuArray([shell.sig_l]);
    case 'Segers'
        s_coeff = shell(1).coeff;
        s_AN    = gpuArray([shell.A_N]);
        s_Am1   = gpuArray([shell.A_m1]);
        s_Am2   = gpuArray([shell.A_m2]);
        s_sigl  = gpuArray([shell.sig_l]);
end

%% Stability: sub-steps per output step
omega_nd = sqrt(3 * max(gather(kap)));
dt_crit  = 2.0 / omega_nd;
n_sub    = max(1, ceil(dt / dt_crit));
h        = dt / n_sub;

fprintf('    [GPU-RK4] N_MB=%d, N_out=%d, n_sub=%d, dt=%.4g, h=%.4g\n', ...
    N_MB, N_out, n_sub, dt, h);

%% Precompute RP equation constants [1 x N_MB]
C1 = gpuArray(P0 * T^2 / rho ./ R0.^2);   % P0*T^2/(rho*R0^2)
C2 = gpuArray(1 + 2*sig0./(R0*P0));        % 1 + 2*sig0/(R0*P0)
C3 = gpuArray(3*kap.*R0 / (c_l*T));        % 3*kappa*R0/(c*T)
C4 = gpuArray(2./(R0*P0));                  % 2/(R0*P0)
C5 = gpuArray(4*nu / (P0*T));              % 4*nu/(P0*T)
C6 = gpuArray(4*Ks./(P0*R0*T));            % 4*Ks/(P0*R0*T)
invP0 = 1 / P0;

%% Initialize state [1 x N_MB]
x  = gpuArray(zeros(1, N_MB));
xd = gpuArray(zeros(1, N_MB));

%% Preallocate output [N_out x N_MB]
X_out  = gpuArray(zeros(N_out, N_MB));
Xd_out = gpuArray(zeros(N_out, N_MB));

%% RK4 integration
for n = 1:(N_out-1)
    Pn  = P_gpu(:, n)';      % [1 x N_MB]
    Pn1 = P_gpu(:, n+1)';

    for s = 0:(n_sub-1)
        a0 = s / n_sub;
        ah = (s + 0.5) / n_sub;
        a1 = (s + 1) / n_sub;

        Ps = Pn + a0 * (Pn1 - Pn);
        Pm = Pn + ah * (Pn1 - Pn);
        Pe = Pn + a1 * (Pn1 - Pn);

        [k1x, k1v] = rp_rhs(x,            xd,            Ps);
        [k2x, k2v] = rp_rhs(x+0.5*h*k1x,  xd+0.5*h*k1v,  Pm);
        [k3x, k3v] = rp_rhs(x+0.5*h*k2x,  xd+0.5*h*k2v,  Pm);
        [k4x, k4v] = rp_rhs(x+h*k3x,      xd+h*k3v,      Pe);

        x  = x  + h/6 * (k1x + 2*k2x + 2*k3x + k4x);
        xd = xd + h/6 * (k1v + 2*k2v + 2*k3v + k4v);
    end

    X_out(n+1,:)  = x;
    Xd_out(n+1,:) = xd;
end

%% Gather and build response struct
X_out  = gather(X_out);
Xd_out = gather(Xd_out);
t_out  = tq(:);

for i = N_MB:-1:1
    response(i).R    = bubble(i).R0 * (1 + X_out(:,i));
    response(i).Rdot = bubble(i).R0 * Xd_out(:,i) / T;
    response(i).t    = t_out;
end

%% ===== Nested RHS function (vectorized across all bubbles) =====
    function [dx, dv] = rp_rhs(xi, xdi, Pi)
        Ri = R0 .* (1 + xi);

        % Surface tension (vectorized)
        switch shell_model
            case 'Marmottant'
                sig = s_chi .* (Ri.^2 ./ s_Rb.^2 - 1);
                sig = max(0, sig);
                sig = min(sig, s_sigl);

            case 'SegersTable'
                Am = 4*pi*Ri.^2 ./ s_AN;
                % Clamp to table domain, then overwrite buckle/rupture
                Am_c = min(max(Am, Am_tbl(1)), Am_tbl(end));
                sig = interp1(Am_tbl, sig_tbl, Am_c, 'linear');
                sig(Am <= s_Am1) = 0;
                sig(Am >= s_Am2) = s_sigl(Am >= s_Am2);

            case 'Segers'
                Am = 4*pi*Ri.^2 ./ s_AN;
                sig = polyval(s_coeff, Am);
                sig(Am < s_Am1) = 0;
                sig(Am > s_Am2) = s_sigl(Am > s_Am2);
        end

        opx = 1 + xi;

        dv = (1./opx) .* (...
            -1.5 .* xdi.^2 ...
            + C1 .* (...
                C2 .* opx.^(-3.*kap) .* (1 - C3.*xdi) ...
                - 1 ...
                - C4 .* sig ./ opx ...
                - C5 .* xdi ./ opx ...
                - C6 .* xdi ./ opx.^2 ...
                - Pi * invP0 ...
            ) ...
        );
        dx = xdi;
    end

end
