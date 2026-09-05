function  Vq = sinc_interpolation(X,V,Xq)
% Performs band-limited resampling of V=F(X) at query points Xq.
% Uses FFT-based resampling for uniform grids (fast O(N log N) path)
% with fallback to spline interpolation.
%
% Original sinc interpolation was O(M*N) due to dense matrix construction.
% This version achieves equivalent accuracy for bandlimited signals at
% dramatically lower cost.

if isrow(V)
    V = V';
end

[N_in, N_sig] = size(V);
N_out = length(Xq);

% Check for uniform input grid (within floating-point tolerance)
dX = diff(X);
uniform_in = (max(dX) - min(dX)) < 1e-10 * max(abs(dX));

dXq = diff(Xq);
uniform_out = (max(dXq) - min(dXq)) < 1e-10 * max(abs(dXq));

if uniform_in && uniform_out && N_in > 1 && N_out > 1
    % The fast path resamples by sample count alone. interpft maps N_in
    % samples onto N_out over the INPUT's period, so the output lands on a
    % grid of step N_in*dX/N_out starting at X(1) -- it never reads Xq. Refuse
    % a query grid that would not answer rather than returning a silently
    % stretched signal, which is what a caller with a different origin or span
    % used to get.
    %
    % The tolerance is deliberately loose. The two callers in
    % compute_bubble_mass_source (k-Wave rate <-> bubble rate) differ by
    % 0.107%, a known and documented stretch that this guard must keep
    % admitting; see docs/implementation_docs/changes_log.md. It is here to
    % catch a grid that is wrong by a lot, not to certify the ones that pass.
    span_in    = N_in  * dX(1);
    span_out   = N_out * dXq(1);
    originSkew = abs(Xq(1) - X(1));
    stretch    = abs(span_out - span_in) / abs(span_in);
    if originSkew > 1e-9 * abs(span_in) || stretch > 0.01
        error('sinc_interpolation:GridMismatch', ...
            ['The FFT resampling path maps %d samples onto %d over the ' ...
             'input period and ignores Xq. Here Xq starts %g after X(1) ' ...
             'and spans %.3g%% off the input, so the result would come ' ...
             'back stretched. Pass grids that share an origin and a span, ' ...
             'or resample explicitly.'], ...
            N_in, N_out, Xq(1) - X(1), 100*stretch);
    end

    % Fast path: FFT-based resampling for uniform grids
    % interpft handles the case where N_out ~= N_in via zero-padding in
    % frequency domain — exact for bandlimited signals.

    % interpft only upsamples; for downsampling we need a different approach
    if N_out >= N_in
        Vq = interpft(V, N_out, 1);
    else
        % Downsample: low-pass filter in frequency domain then truncate
        V_fft = fft(V, [], 1);
        Vq_fft = zeros(N_out, N_sig, class(V));

        % Copy positive frequencies
        n_pos = floor(N_out / 2);
        Vq_fft(1:n_pos+1, :) = V_fft(1:n_pos+1, :);

        % Copy negative frequencies
        n_neg = N_out - n_pos - 1;
        if n_neg > 0
            Vq_fft(end-n_neg+1:end, :) = V_fft(end-n_neg+1:end, :);
        end

        % Handle Nyquist bin for even-length outputs. A length-N_out DFT has
        % a single bin there, so it has to carry both halves of the input's
        % conjugate pair: the +N_out/2 bin copied above and the -N_out/2 bin,
        % which the negative-frequency copy deliberately leaves out. For real
        % V those sum to 2*real(V_fft(+N_out/2)). Taking only the real part
        % returned a component sitting exactly on the new Nyquist at half
        % amplitude, and cost the up-then-down round trip its exactness.
        if mod(N_out, 2) == 0
            Vq_fft(n_pos+1, :) = 2 * real(Vq_fft(n_pos+1, :));
        end

        Vq = real(ifft(Vq_fft, [], 1)) * (N_out / N_in);
    end
else
    % Fallback: spline interpolation (still much faster than dense sinc)
    Vq = interp1(X(:), V, Xq(:), 'spline');
end

Vq = Vq';

end
