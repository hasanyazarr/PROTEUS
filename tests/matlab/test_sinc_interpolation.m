repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));

% sinc_interpolation returns [N_sig x N_out]; every signal below is a single
% column in, a single row out.

%% A component at exactly the new Nyquist keeps its amplitude.
% N_in = 900 -> N_out = 300 puts bin 150 of the input on the output's Nyquist.
% The old code copied only the +150 bin and dropped its conjugate, so a cosine
% there came back at half amplitude.
N_in  = 900;
N_out = 300;
n     = (0:N_in-1)';
X     = n;                       % dX = 1
Xq    = (0:N_out-1)' * (N_in/N_out);
v     = cos(2*pi*150*n/N_in);    % exactly the output Nyquist frequency
vq    = sinc_interpolation(X, v, Xq);
assert(isequal(size(vq), [1 N_out]), 'orientation changed');
assert(abs(max(abs(vq)) - 1) < 1e-10, ...
    'Nyquist cosine came back at %.6f, expected 1', max(abs(vq)));

%% The bin below Nyquist is untouched by that correction.
v   = cos(2*pi*149*n/N_in);
vq  = sinc_interpolation(X, v, Xq);
assert(abs(max(abs(vq)) - 1) < 1e-10, ...
    'sub-Nyquist cosine came back at %.6f, expected 1', max(abs(vq)));

%% An odd-length downsample has no Nyquist bin and must not be scaled.
N_out_odd = 225;
Xq_odd    = (0:N_out_odd-1)' * (N_in/N_out_odd);
v         = cos(2*pi*100*n/N_in);
vq        = sinc_interpolation(X, v, Xq_odd);
assert(isequal(size(vq), [1 N_out_odd]));
assert(abs(max(abs(vq)) - 1) < 1e-10, ...
    'odd-length downsample changed amplitude: %.6f', max(abs(vq)));

%% Up then down is the identity, including at the Nyquist bin.
% This is the shape compute_bubble_mass_source uses: k-Wave rate -> bubble
% rate -> k-Wave rate. The signal here is full-band on purpose, so the input's
% Nyquist bin is non-zero and the round trip only closes if the downsample
% restores what interpft split.
rng(0);
N_up = 6811;
x    = randn(N_in, 3);
X_up = (0:N_up-1)' * (N_in/N_up);
up   = sinc_interpolation(X, x, X_up);          % [3 x N_up]
back = sinc_interpolation(X_up, transpose(up), X);  % [3 x N_in]
err  = max(abs(transpose(back) - x), [], 'all') / max(abs(x), [], 'all');
assert(err < 1e-12, 'round trip lost %.3e relative', err);

%% The N_out/N_in scaling keeps DC intact.
% A constant is the one signal whose amplitude the scale factor alone decides,
% so this is the guard against a regression in that factor. (rms is NOT
% preserved for a tone sitting exactly on the Nyquist: its decimated samples
% land on the peaks, which is a property of Nyquist sampling, not a defect.)
v  = 3.25 * ones(N_in, 1);
vq = sinc_interpolation(X, v, Xq);
assert(max(abs(vq - 3.25)) < 1e-10, 'DC drifted to %.6f', vq(1));

%% Energy is preserved for a tone below the new Nyquist.
v  = cos(2*pi*100*n/N_in);          % 100 cycles -> integer cycles after decim.
vq = sinc_interpolation(X, v, Xq);
assert(abs(rms(vq) - rms(v)) < 1e-10, 'rms %.6f -> %.6f', rms(v), rms(vq));

%% The fast path refuses a grid it cannot answer, instead of stretching it.
% interpft maps N_in samples onto N_out over the INPUT's period and never
% reads Xq, so a query grid with a different origin or a materially different
% span would come back silently wrong. See the guard's comment for why the
% tolerance is loose rather than exact.
shifted = Xq + 500;                                  % same step, wrong origin
assert_errors(@() sinc_interpolation(X, cos(2*pi*10*n/N_in), shifted), ...
    'sinc_interpolation:GridMismatch', 'shifted origin');

halfSpan = (0:N_out-1)' * (N_in/N_out) / 2;          % covers half the record
assert_errors(@() sinc_interpolation(X, cos(2*pi*10*n/N_in), halfSpan), ...
    'sinc_interpolation:GridMismatch', 'half span');

%% The pair compute_bubble_mass_source actually uses stays on the fast path.
% k-Wave 900 samples at 33 MHz -> bubble module at 250 MHz. The two grids
% differ by 0.107%, which is the known, documented stretch: accepted here.
fs_kwave = 33e6;
fs_MB    = 250e6;
t_kwave  = (0:899)' / fs_kwave;
M        = floor(t_kwave(end)*fs_MB) + 1;
t_MB     = (0:M-1)' / fs_MB;
assert(M == 6811, 'representative case changed shape: M = %d', M);
up = sinc_interpolation(t_kwave, cos(2*pi*5e6*t_kwave), t_MB);
assert(isequal(size(up), [1 M]), 'production-shaped call left the fast path');
down = sinc_interpolation(t_MB, transpose(up), t_kwave);
assert(isequal(size(down), [1 900]));

%% A non-uniform query grid still falls through to spline, no error raised.
irregular = sort(rand(50,1)) * (N_in-1);
vq = sinc_interpolation(X, cos(2*pi*10*n/N_in), irregular);
assert(isequal(size(vq), [1 50]), 'spline fallback changed shape');

disp('test_sinc_interpolation: OK');


function assert_errors(fn, expectedId, what)
try
    fn();
catch err
    assert(strcmp(err.identifier, expectedId), ...
        '%s raised %s, expected %s', what, err.identifier, expectedId);
    return
end
error('%s did not raise %s', what, expectedId);
end
