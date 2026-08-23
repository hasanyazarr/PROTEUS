function [W_rise, W_lo, W_hi] = hermite_stage_weights(fracs, method)
% Basis weights that evaluate the pressure at normalised positions FRACS
% inside a coarse interval, given the rise across it and the two scaled
% endpoint slopes.
%
% Linear:  P(s) = P_n + s*dP
% Pchip:   P(s) = P_n + (3s^2 - 2s^3)*dP + (s^3 - 2s^2 + s)*H*m_n
%                     + (s^3 - s^2)*H*m_{n+1}
% which is the standard cubic Hermite form rewritten around P_n.

if strcmp(method, 'linear')
    W_rise = fracs;
    W_lo = zeros(size(fracs), 'like', fracs);
    W_hi = W_lo;
    return
end

W_rise = fracs.^2 .* (3 - 2*fracs);
W_lo   = fracs .* (fracs - 1).^2;
W_hi   = fracs.^2 .* (fracs - 1);
end

