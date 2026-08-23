function m = pchip_slopes(x, Y)
% Shape-preserving (Fritsch-Carlson) slopes at the knots X for every row of Y.
% Same construction MATLAB's pchip uses, vectorised over the rows so the whole
% microbubble population is handled in one pass.

x = x(:)';
n = numel(x);
h = diff(x);                       % [1 x n-1]
delta = diff(Y, 1, 2) ./ h;        % [rows x n-1]
m = zeros(size(Y), 'like', Y);

if n == 2
    m = repmat(delta, 1, 2);
    return
end

% Interior knots: weighted harmonic mean, zeroed at sign changes and extrema.
hL = h(1:end-1);
hR = h(2:end);
dL = delta(:, 1:end-1);
dR = delta(:, 2:end);
w1 = 2*hR + hL;
w2 = hR + 2*hL;
sameSign = (sign(dL) .* sign(dR)) > 0;
denom = w1 ./ dL + w2 ./ dR;       % harmless Inf/NaN where sameSign is false
interior = (w1 + w2) ./ denom;
interior(~sameSign) = 0;
m(:, 2:end-1) = interior;

% One-sided three-point ends, clipped so they cannot introduce an overshoot.
m(:, 1)   = pchip_end_slope(h(1), h(2), delta(:, 1), delta(:, 2));
m(:, end) = pchip_end_slope(h(end), h(end-1), delta(:, end), delta(:, end-1));
end

function d = pchip_end_slope(h1, h2, del1, del2)
% Non-centred three-point slope estimate at an interval end, limited the way
% MATLAB's pchip limits it.

d = ((2*h1 + h2) .* del1 - h1 .* del2) ./ (h1 + h2);
flip = sign(d) ~= sign(del1);
d(flip) = 0;
clip = ~flip & (sign(del1) ~= sign(del2)) & (abs(d) > abs(3*del1));
d(clip) = 3*del1(clip);
end
