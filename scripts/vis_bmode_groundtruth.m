% visualize_bmode_gt.m
% Contrast-enhanced B-mode: SVD clutter-filtered (bubble-only) image in
% grayscale with ground truth microbubble positions overlaid as blue *.
%
% This matches the reference style: dark background, bright white bubble
% echoes, blue star markers at GT positions.
%
% Usage:
%   >> path_setup
%   >> visualize_bmode_gt

%==========================================================================
% CONFIGURE
%==========================================================================
PATHS = path_setup();
addpath(PATHS.Start);
addpath(fullfile(PATHS.Start, 'delay-and-sum'));
addpath(fullfile(PATHS.Start, 'acoustic-module'));
addpath(PATHS.GUIfunctions);
addpath(PATHS.StreamlineFunctions);

RUN_FOLDER = 'run_20260220_012057';   % <-- change this to your run folder

SETTINGS_NAME = 'GUI_output_parameters_v2.mat';
RESULTS_FOLDER = fullfile(PATHS.ResultsPath, RUN_FOLDER);
GT_FOLDER = fullfile(PATHS.GroundTruthPath, RUN_FOLDER);

% Use settings copy inside the run folder
SETTINGS_PATH = fullfile(RESULTS_FOLDER, SETTINGS_NAME);
if ~exist(SETTINGS_PATH, 'file')
    SETTINGS_PATH = fullfile(PATHS.SettingsPath, SETTINGS_NAME);
    fprintf('Note: Using settings from %s\n', SETTINGS_PATH);
end

% Output subfolder
VIS_FOLDER = fullfile(RESULTS_FOLDER, 'bmode_gt_vis');
if ~exist(VIS_FOLDER, 'dir'), mkdir(VIS_FOLDER); end

% SVD clutter filter: remove first N singular values (tissue)
SVD_CUTOFF = 2;   % 1-2 is usually enough (first SV dominates clutter)

dynamicRange = 40; % dB

%==========================================================================
% LOAD SETTINGS
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

%==========================================================================
% LOAD ALL RF DATA
%==========================================================================
fprintf('=== Loading RF data ===\n');
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% SVD CLUTTER FILTER
%==========================================================================
fprintf('=== SVD clutter filter (removing first %d singular values) ===\n', SVD_CUTOFF);

RF_casorati = double(reshape(RF, [Nelem*Nt, Nframes]));
[U, S, V] = svd(RF_casorati, 'econ');

% Zero out clutter components
S_filt = S;
for i = 1:min(SVD_CUTOFF, size(S,1))
    S_filt(i,i) = 0;
end
RF_filtered = U * S_filt * V';
RF_filtered = reshape(RF_filtered, [Nelem, Nt, Nframes]);
RF_filtered = single(RF_filtered);

fprintf('  Original RF range: [%.2e, %.2e]\n', min(RF(:)), max(RF(:)));
fprintf('  Filtered RF range: [%.2e, %.2e]\n', min(RF_filtered(:)), max(RF_filtered(:)));

%==========================================================================
% DAS RECONSTRUCTION ON FILTERED DATA
%==========================================================================
fprintf('=== DAS reconstruction on bubble-only data ===\n');

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
x_lat = -width/2:pixelSize:width/2;
z_ax  = 0:pixelSize:depth;

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
M_DAS = compute_das_matrix(t, x_lat, z_ax, x_el, c, Fs, focus);

% Apply DAS
fprintf('Applying DAS matrix...\n');
RF_das = permute(RF_filtered, [2 1 3]);  % Nt x Nelem x Nframes
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);

IMG = full(M_DAS * RF_das);
IMG = reshape(IMG, [length(x_lat), length(z_ax), Nframes]);

% Envelope detection + log compression
IMG = abs(IMG);
IMG_global_max = max(IMG(:));
if IMG_global_max <= 0, IMG_global_max = 1; end
IMG_db = 20*log10(IMG / IMG_global_max);

fprintf('  IMG range: [%.2f, %.2f] dB\n', min(IMG_db(:)), max(IMG_db(:)));

% Transpose so rows=axial(z), cols=lateral(x)
IMG_display = permute(IMG_db, [2 1 3]);
IMG_display(isnan(IMG_display)) = -dynamicRange;
IMG_display(IMG_display < -dynamicRange) = -dynamicRange;

%==========================================================================
% GROUND TRUTH SETUP
%==========================================================================
NFrames_total = Acquisition.NumberOfFrames;
npad_gt = length(num2str(NFrames_total));
pulse_name = 'Pulse1';

fsp_file = fullfile(GT_FOLDER, 'FlowSimulationParameters.mat');
if exist(fsp_file, 'file')
    fsp = load(fsp_file, 'FlowSimulationParameters');
    frameRate = fsp.FlowSimulationParameters.FrameRate;
else
    frameRate = Acquisition.FrameRate;
end

%==========================================================================
% SAVE FRAMES
%==========================================================================
fprintf('=== Saving contrast B-mode + GT overlay frames ===\n');
npad = length(num2str(Nframes));

for iframe = 1:Nframes
    gt = load_gt_frame(GT_FOLDER, iframe, npad_gt, pulse_name, Geometry);

    fig = figure('Visible', 'off', 'Color', 'k', 'InvertHardcopy', 'off', ...
        'Position', [100 100 500 800]);
    ax = axes(fig, 'Color', 'k');

    imagesc(ax, x_lat*1e3, z_ax*1e3, IMG_display(:,:,iframe), [-dynamicRange 0]);
    colormap(ax, gray);
    axis(ax, 'image');
    hold(ax, 'on');

    if ~isempty(gt)
        plot(ax, gt(:,1), gt(:,2), '*', 'Color', [0.3 0.6 1], ...
            'MarkerSize', 6, 'LineWidth', 1);
    end
    hold(ax, 'off');

    xlabel(ax, 'Width [mm]', 'Color', 'w');
    ylabel(ax, 'Depth [mm]', 'Color', 'w');
    title(ax, sprintf('%s - Frame rate %dHz', RUN_FOLDER, round(frameRate)), ...
        'Color', 'w', 'Interpreter', 'none');
    ax.XColor = 'w';
    ax.YColor = 'w';

    fname = fullfile(VIS_FOLDER, sprintf('BmodeGT_Frame_%s.png', ...
        num2str(iframe, ['%0' num2str(npad) 'd'])));
    exportgraphics(fig, fname, 'BackgroundColor', 'k', 'Resolution', 150);
    close(fig);

    if mod(iframe, 10) == 0 || iframe == Nframes || iframe == 1
        fprintf('  Saved %d / %d\n', iframe, Nframes);
    end
end
fprintf('Saved to: %s\n', VIS_FOLDER);

%==========================================================================
% INTERACTIVE SLIDER
%==========================================================================
gt1 = load_gt_frame(GT_FOLDER, 1, npad_gt, pulse_name, Geometry);

fig_browse = figure('Name', 'Contrast B-mode + Ground Truth', 'Color', 'k', ...
    'Position', [100 100 500 800]);
ax = axes(fig_browse, 'Color', 'k');
h_img = imagesc(ax, x_lat*1e3, z_ax*1e3, IMG_display(:,:,1), [-dynamicRange 0]);
colormap(ax, gray);
axis(ax, 'image');
hold(ax, 'on');
h_gt = plot(ax, gt1(:,1), gt1(:,2), '*', 'Color', [0.3 0.6 1], ...
    'MarkerSize', 6, 'LineWidth', 1);
hold(ax, 'off');
xlabel(ax, 'Width [mm]', 'Color', 'w');
ylabel(ax, 'Depth [mm]', 'Color', 'w');
title(ax, sprintf('%s - Frame 1 / %d  (FR=%dHz, SVD=%d)', ...
    RUN_FOLDER, Nframes, round(frameRate), SVD_CUTOFF), ...
    'Color', 'w', 'Interpreter', 'none');
ax.XColor = 'w';
ax.YColor = 'w';

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
        gt_k = load_gt_frame(GT_FOLDER, k, npad_gt, pulse_name, Geometry);
        set(h_gt, 'XData', gt_k(:,1), 'YData', gt_k(:,2));
        title(ax, sprintf('%s - Frame %d / %d  (FR=%dHz, SVD=%d)', ...
            RUN_FOLDER, k, Nframes, round(frameRate), SVD_CUTOFF), ...
            'Color', 'w', 'Interpreter', 'none');
    end

% --- Sample grid ---
Ngrid = min(20, Nframes);
idx_grid = round(linspace(1, Nframes, Ngrid));
fig_grid = figure('Name', 'Contrast B-mode + GT grid', 'Color', 'k', ...
    'Position', [50 50 1400 800]);
[ny, nx_g] = subplot_grid(Ngrid);
for i = 1:Ngrid
    subplot(ny, nx_g, i);
    imagesc(x_lat*1e3, z_ax*1e3, IMG_display(:,:,idx_grid(i)), [-dynamicRange 0]);
    colormap(gca, gray);
    axis image;
    hold on;
    gt_i = load_gt_frame(GT_FOLDER, idx_grid(i), npad_gt, pulse_name, Geometry);
    plot(gt_i(:,1), gt_i(:,2), '*', 'Color', [0.3 0.6 1], 'MarkerSize', 3);
    hold off;
    title(sprintf('F%d', idx_grid(i)), 'Color', 'w');
    set(gca, 'XColor', 'w', 'YColor', 'w', 'Color', 'k');
end
sgtitle(sprintf('%s  (%d frames, SVD cutoff=%d)', RUN_FOLDER, Nframes, SVD_CUTOFF), ...
    'Color', 'w', 'Interpreter', 'none');
set(fig_grid, 'InvertHardcopy', 'off');
saveas(fig_grid, fullfile(VIS_FOLDER, 'BmodeGT_Sample_Grid.png'));

fprintf('=== Visualization complete ===\n');
fprintf('All outputs saved to: %s\n', VIS_FOLDER);

%==========================================================================
% HELPER FUNCTIONS
%==========================================================================
function gt_mm = load_gt_frame(gt_folder, frame_idx, npad, pulse_name, Geom)
    gt_file = fullfile(gt_folder, sprintf('Frame_%s.mat', ...
        num2str(frame_idx, ['%0' num2str(npad) 'd'])));
    gt_mm = zeros(0, 2);
    if ~exist(gt_file, 'file'), return; end
    gt_data = load(gt_file, 'Frame');
    if ~isfield(gt_data.Frame, pulse_name), return; end
    pts_raw = gt_data.Frame.(pulse_name).Points;
    pts = pts_raw';
    pts = pts - Geom.BoundingBox.Center;
    pts = Geom.Rotation * pts;
    pts = pts + Geom.Center;
    pts = pts';
    gt_mm = [pts(:,2)*1e3, pts(:,1)*1e3]; % [lateral_mm, axial_mm]
end

function [ny, nx] = subplot_grid(N)
    nx = ceil(sqrt(N));
    ny = ceil(N / nx);
end
