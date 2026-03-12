% visualize_microbubbles.m
% Isolate microbubble echoes from tissue clutter using SVD filtering,
% then display as white dots on a black background.
%
% SVD clutter filter: The tissue signal is static across frames, so it
% occupies the first few singular values. Removing them isolates the
% time-varying microbubble echoes.
%
% Usage:
%   >> path_setup
%   >> visualize_microbubbles

%==========================================================================
% SETTINGS
%==========================================================================
PATHS = path_setup();
addpath(PATHS.Start);
addpath(fullfile(PATHS.Start, 'delay-and-sum'));
addpath(PATHS.GUIfunctions);

%==========================================================================
% CONFIGURE: set this to your run folder name
%==========================================================================
RUN_FOLDER = 'run_20260220_012057';   % <-- change this to your run folder

SETTINGS_NAME = 'GUI_output_parameters_v2.mat';
RESULTS_FOLDER = fullfile(PATHS.ResultsPath, RUN_FOLDER);

% Use the settings copy saved inside the run folder
SETTINGS_PATH = fullfile(RESULTS_FOLDER, SETTINGS_NAME);
if ~exist(SETTINGS_PATH, 'file')
    SETTINGS_PATH = fullfile(PATHS.SettingsPath, SETTINGS_NAME);
    fprintf('Note: Using settings from %s\n', SETTINGS_PATH);
end

% Output subfolder inside the run
VIS_FOLDER = fullfile(RESULTS_FOLDER, 'microbubble_vis');
if ~exist(VIS_FOLDER, 'dir'), mkdir(VIS_FOLDER); end
fprintf('Microbubble visualizations will be saved to: %s\n', VIS_FOLDER);

% SVD clutter filter: remove singular values 1..SVD_CUTOFF (tissue clutter)
SVD_CUTOFF = 20;   % tune this: higher = more clutter removed (try 5-30)

dynamicRange = 30; % dB dynamic range for display

if ~exist(RESULTS_FOLDER, 'dir')
    error('Results folder not found: %s', RESULTS_FOLDER);
end

%==========================================================================
% LOAD SETTINGS
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'SimulationParameters', ...
    'Transmit', 'Transducer', 'Geometry', 'Medium');

%==========================================================================
% LOAD ALL RF DATA
%==========================================================================
fprintf('=== Loading RF data ===\n');
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
% RF: (Nelem x Nt x Nframes)
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% SVD CLUTTER FILTER
%==========================================================================
fprintf('=== Applying SVD clutter filter (removing first %d singular values) ===\n', SVD_CUTOFF);

% Reshape to Casorati matrix: (spatial x temporal)
% spatial = Nelem*Nt, temporal = Nframes
RF_casorati = double(reshape(RF, [Nelem*Nt, Nframes]));

% SVD decomposition
[U, S, V] = svd(RF_casorati, 'econ');

% Show singular value spectrum
figure('Name', 'SVD Spectrum');
semilogy(diag(S), 'b.-', 'MarkerSize', 10);
hold on;
xline(SVD_CUTOFF + 0.5, 'r--', 'LineWidth', 2);
xlabel('Singular Value Index');
ylabel('Singular Value');
title('SVD Spectrum (red line = cutoff)');
legend('Singular values', sprintf('Cutoff = %d', SVD_CUTOFF));
grid on;

% Zero out first SVD_CUTOFF singular values (tissue clutter)
S_filtered = S;
for i = 1:min(SVD_CUTOFF, size(S,1))
    S_filtered(i,i) = 0;
end

% Reconstruct filtered RF
RF_filtered = U * S_filtered * V';
RF_filtered = reshape(RF_filtered, [Nelem, Nt, Nframes]);
RF_filtered = single(RF_filtered);

fprintf('  Original RF range: [%.2e, %.2e]\n', min(RF(:)), max(RF(:)));
fprintf('  Filtered RF range: [%.2e, %.2e]\n', min(RF_filtered(:)), max(RF_filtered(:)));

%==========================================================================
% DAS RECONSTRUCTION ON FILTERED DATA
%==========================================================================
fprintf('=== DAS reconstruction on microbubble-only data ===\n');

Fs = SimulationParameters.SamplingRate;
t = (0:(Nt-1))/Fs;

% Transducer geometry
p = Transducer.Pitch;
x_el = -p/2*(Nelem-1) + (0:(Nelem-1))*p;
focus = Transmit.LateralFocus;

% Reconstruction grid
c  = Medium.SpeedOfSound;
f0 = Transmit.CenterFrequency;
lambda = c / f0;
pixelSize = lambda / 5;

Domain = Geometry.Domain;
width = Domain.Ymax - Domain.Ymin;
depth = Domain.Xmax;
x = -width/2:pixelSize:width/2;
z = 0:pixelSize:depth;

% Time corrections (same as DAS_reconstruction.m)
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

% TGC
att = Medium.AttenuationA * (f0*1e-6)^Medium.AttenuationB;
TGC = sqrt(t) / max(sqrt(t)) .* 10.^(att.*t.*c.*1e2./20./2);
TGC(t<0) = 0;
RF_filtered = RF_filtered .* reshape(TGC, [1, Nt, 1]);

% Compute DAS matrix
fprintf('Computing DAS matrix...\n');
M_DAS = compute_das_matrix(t, x, z, x_el, c, Fs, focus);

% Apply DAS
fprintf('Applying DAS matrix...\n');
RF_das = permute(RF_filtered, [2 1 3]);  % Nt x Nelem x Nframes
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);

IMG = full(M_DAS * RF_das);
IMG = reshape(IMG, [length(x), length(z), Nframes]);

% Envelope detection
IMG = abs(IMG);

% Normalize globally (so bubble brightness is consistent across frames)
IMG_global_max = max(IMG(:));
if IMG_global_max <= 0
    IMG_global_max = 1;
end
IMG_db = 20*log10(IMG / IMG_global_max);

fprintf('  IMG range: [%.2f, %.2f] dB\n', min(IMG_db(:)), max(IMG_db(:)));

%==========================================================================
% DISPLAY: WHITE BUBBLES ON BLACK BACKGROUND
%==========================================================================
% Transpose so rows=axial(z), cols=lateral(x)
IMG_display = permute(IMG_db, [2 1 3]);
IMG_display(isnan(IMG_display)) = -dynamicRange;
IMG_display(IMG_display < -dynamicRange) = -dynamicRange;

% 1) Save PNGs
fprintf('=== Saving microbubble B-mode PNGs ===\n');
npad = length(num2str(Nframes));
for iframe = 1:Nframes
    fname = fullfile(VIS_FOLDER, ...
        sprintf('MB_Frame_%s.png', num2str(iframe, ['%0' num2str(npad) 'd'])));
    fig_vis = figure('Visible', 'off');
    imagesc(x*1e3, z*1e3, IMG_display(:,:,iframe), [-dynamicRange 0]);
    colormap(hot);  % hot colormap: black background, bright bubbles
    axis image;
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    title(sprintf('Microbubbles - Frame %d', iframe));
    colorbar;
    saveas(fig_vis, fname);
    close(fig_vis);
    if mod(iframe, 20) == 0 || iframe == Nframes
        fprintf('  Saved %d / %d\n', iframe, Nframes);
    end
end
fprintf('Saved to: %s\n', VIS_FOLDER);

% Also save SVD spectrum
saveas(findobj('Name', 'SVD Spectrum'), fullfile(VIS_FOLDER, 'SVD_Spectrum.png'));

% 2) Interactive slider
fig_browse = figure('Name', 'Microbubble Echoes (SVD filtered)');
ax = axes(fig_browse);
h_img = imagesc(ax, x*1e3, z*1e3, IMG_display(:,:,1), [-dynamicRange 0]);
colormap(ax, hot);
axis(ax, 'image');
xlabel(ax, 'Lateral (mm)');
ylabel(ax, 'Axial (mm)');
title(ax, sprintf('Microbubbles - Frame 1 / %d  (SVD cutoff=%d)', Nframes, SVD_CUTOFF));
colorbar(ax);

if Nframes > 1
    h_slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', Nframes, 'Value', 1, ...
        'Position', [20 10 400 20], ...
        'SliderStep', [1/max(1,Nframes-1) 10/max(1,Nframes-1)], ...
        'Callback', @(~,~) updateFrame());
    uicontrol('Style', 'text', 'Position', [430 10 80 20], 'String', 'Frame');
end

function updateFrame()
if Nframes == 1, return; end
k = round(get(h_slider, 'Value'));
k = max(1, min(Nframes, k));
set(h_img, 'CData', IMG_display(:,:,k));
title(ax, sprintf('Microbubbles - Frame %d / %d  (SVD cutoff=%d)', k, Nframes, SVD_CUTOFF));
end

% 3) Side-by-side: tissue B-mode vs microbubble-only (frame 1)
fprintf('=== Comparison: Tissue vs Microbubble ===\n');
[IMG_tissue, ~, ~] = DAS_reconstruction(RESULTS_FOLDER, SETTINGS_PATH);
IMG_tissue_disp = permute(IMG_tissue, [2 1 3]);
IMG_tissue_disp(isnan(IMG_tissue_disp)) = -60;
IMG_tissue_disp(IMG_tissue_disp < -60) = -60;

mid_frame = round(Nframes/2);
figure('Name', 'Comparison: Tissue vs Microbubbles', 'Position', [50 50 1200 500]);

subplot(1,2,1);
imagesc(x*1e3, z*1e3, IMG_tissue_disp(:,:,mid_frame), [-60 0]);
colormap(gca, gray);
axis image;
xlabel('Lateral (mm)');
ylabel('Axial (mm)');
title(sprintf('Standard B-mode (Frame %d)', mid_frame));
colorbar;

subplot(1,2,2);
imagesc(x*1e3, z*1e3, IMG_display(:,:,mid_frame), [-dynamicRange 0]);
colormap(gca, hot);
axis image;
xlabel('Lateral (mm)');
ylabel('Axial (mm)');
title(sprintf('Microbubbles Only (Frame %d, SVD cutoff=%d)', mid_frame, SVD_CUTOFF));
colorbar;

sgtitle('Tissue Clutter vs Microbubble Echoes');
saveas(gcf, fullfile(VIS_FOLDER, 'Comparison_Tissue_vs_Microbubbles.png'));

% 4) Sample grid
Ngrid = min(20, Nframes);
idx_grid = round(linspace(1, Nframes, Ngrid));
figure('Name', 'Microbubble frames sample grid');
[ny_g, nx_g] = subplot_grid(Ngrid);
for i = 1:Ngrid
    subplot(ny_g, nx_g, i);
    imagesc(x*1e3, z*1e3, IMG_display(:,:,idx_grid(i)), [-dynamicRange 0]);
    colormap(gca, hot);
    axis image;
    title(sprintf('F%d', idx_grid(i)));
end
sgtitle(sprintf('Microbubble Echoes (%d frames, SVD cutoff=%d)', Nframes, SVD_CUTOFF));
saveas(gcf, fullfile(VIS_FOLDER, 'Microbubble_Sample_Grid.png'));

fprintf('=== Visualization complete ===\n');
fprintf('All visualizations saved to: %s\n', VIS_FOLDER);
fprintf('Tip: Adjust SVD_CUTOFF (currently %d) if too much clutter or too little signal.\n', SVD_CUTOFF);
fprintf('  Lower = more tissue visible; Higher = cleaner but may lose weak bubbles.\n');

function [ny, nx] = subplot_grid(N)
nx = ceil(sqrt(N));
ny = ceil(N / nx);
end
