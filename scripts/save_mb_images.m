% save_mb_images.m
% Save raw microbubble-only images (no axes, no markers, no titles).
% Black background, white bubble echoes. Pure image output for ML/DL.
%
% Usage:
%   >> path_setup
%   >> save_mb_images

%==========================================================================
% CONFIGURE
%==========================================================================
PATHS = path_setup();
addpath(PATHS.Start);
addpath(fullfile(PATHS.Start, 'delay-and-sum'));
addpath(PATHS.GUIfunctions);

RUN_FOLDER = 'run_20260220_012057';   % <-- change this to your run folder

SETTINGS_NAME = 'GUI_output_parameters_v2.mat';
RESULTS_FOLDER = fullfile(PATHS.ResultsPath, RUN_FOLDER);

SETTINGS_PATH = fullfile(RESULTS_FOLDER, SETTINGS_NAME);
if ~exist(SETTINGS_PATH, 'file')
    SETTINGS_PATH = fullfile(PATHS.SettingsPath, SETTINGS_NAME);
end

% Output subfolder
VIS_FOLDER = fullfile(RESULTS_FOLDER, 'mb_images_raw');
if ~exist(VIS_FOLDER, 'dir'), mkdir(VIS_FOLDER); end

SVD_CUTOFF = 2;
dynamicRange = 40; % dB

%==========================================================================
% LOAD SETTINGS & RF DATA
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

fprintf('=== Loading RF data ===\n');
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% SVD CLUTTER FILTER
%==========================================================================
fprintf('=== SVD clutter filter (cutoff=%d) ===\n', SVD_CUTOFF);
RF_casorati = double(reshape(RF, [Nelem*Nt, Nframes]));
[U, S, V] = svd(RF_casorati, 'econ');
for i = 1:min(SVD_CUTOFF, size(S,1))
    S(i,i) = 0;
end
RF_filtered = single(reshape(U * S * V', [Nelem, Nt, Nframes]));

%==========================================================================
% DAS RECONSTRUCTION
%==========================================================================
fprintf('=== DAS reconstruction ===\n');
Fs = SimulationParameters.SamplingRate;
t = (0:(Nt-1))/Fs;

p = Transducer.Pitch;
x_el = -p/2*(Nelem-1) + (0:(Nelem-1))*p;
focus = Transmit.LateralFocus;

c  = Medium.SpeedOfSound;
f0 = Transmit.CenterFrequency;
lambda = c / f0;
pixelSize = lambda / 5;

Domain = Geometry.Domain;
width = Domain.Ymax - Domain.Ymin;
depth = Domain.Xmax;
x_lat = -width/2:pixelSize:width/2;
z_ax  = 0:pixelSize:depth;

IR = Transducer.ReceiveImpulseResponse;
V_ref = conv(Transmit.PressureSignal, IR) / Transducer.SamplingRate;
[~, I] = max(abs(hilbert(V_ref)));
timeToPeak = I / Transmit.SamplingRate;

H = Transducer.ElementHeight;
F = Transducer.ElevationFocus;
if isfinite(abs(F))
    lensCorrection = sqrt((H/2)^2 + F^2)/c - F/c;
else
    lensCorrection = 0;
end
lensCorrection = 2 * lensCorrection;

dt_sim = 1 / SimulationParameters.SamplingRate;
dx_sim = SimulationParameters.GridSize;
if SimulationParameters.HybridSimulation
    kWaveCorrection = dx_sim/(2*c) + dt_sim;
else
    kWaveCorrection = dx_sim/(2*c) + dt_sim*3/2;
end
t = t - timeToPeak - lensCorrection + kWaveCorrection;

att = Medium.AttenuationA * (f0*1e-6)^Medium.AttenuationB;
TGC = sqrt(t) / max(sqrt(t)) .* 10.^(att.*t.*c.*1e2./20./2);
TGC(t<0) = 0;
RF_filtered = RF_filtered .* reshape(TGC, [1, Nt, 1]);

fprintf('Computing DAS matrix...\n');
M_DAS = compute_das_matrix(t, x_lat, z_ax, x_el, c, Fs, focus);

fprintf('Applying DAS matrix...\n');
RF_das = permute(RF_filtered, [2 1 3]);
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);

IMG = full(M_DAS * RF_das);
IMG = reshape(IMG, [length(x_lat), length(z_ax), Nframes]);

IMG = abs(IMG);
IMG_global_max = max(IMG(:));
if IMG_global_max <= 0, IMG_global_max = 1; end
IMG_db = 20*log10(IMG / IMG_global_max);

% Transpose: rows=axial, cols=lateral
IMG_display = permute(IMG_db, [2 1 3]);
IMG_display(isnan(IMG_display)) = -dynamicRange;
IMG_display(IMG_display < -dynamicRange) = -dynamicRange;

% Normalize to uint8 [0..255] for saving (0=black, 255=white)
% Per-frame normalization so each frame uses full dynamic range
IMG_uint8 = zeros(size(IMG_display), 'uint8');
for f = 1:Nframes
    frame_f = IMG_display(:,:,f);
    fmin = min(frame_f(:));
    fmax = max(frame_f(:));
    if fmax > fmin
        frame_norm = (frame_f - fmin) / (fmax - fmin);
    else
        frame_norm = zeros(size(frame_f));
    end
    IMG_uint8(:,:,f) = uint8(round(frame_norm * 255));
end

%==========================================================================
% SAVE RAW IMAGES (no axes, no markers, no borders)
%==========================================================================
fprintf('=== Saving raw MB images ===\n');
npad = length(num2str(Nframes));

for iframe = 1:Nframes
    fname = fullfile(VIS_FOLDER, sprintf('MB_%s.png', ...
        num2str(iframe, ['%0' num2str(npad) 'd'])));
    imwrite(IMG_uint8(:,:,iframe), fname);

    if mod(iframe, 20) == 0 || iframe == Nframes || iframe == 1
        fprintf('  Saved %d / %d\n', iframe, Nframes);
    end
end

fprintf('=== Done. %d images saved to: %s ===\n', Nframes, VIS_FOLDER);
fprintf('Image size: %d x %d pixels\n', size(IMG_norm,1), size(IMG_norm,2));
