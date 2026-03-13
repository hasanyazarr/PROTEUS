function visualize_all(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER)
% visualize_all  Save sample frames from 3 visualization modes:
%   1) B-mode (grayscale)
%   2) Microbubble-only (SVD filtered, hot colormap)
%   3) B-mode + Ground Truth overlay
%
% Usage:
%   visualize_all(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER)

if ~exist(OUT_FOLDER, 'dir'), mkdir(OUT_FOLDER); end

%==========================================================================
% LOAD SETTINGS & RF DATA
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

fprintf('=== Loading RF data from %s ===\n', RESULTS_FOLDER);
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% COMMON DAS SETUP
%==========================================================================
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

% Time corrections
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

% DAS matrix (compute once, reuse for all modes)
fprintf('Computing DAS matrix...\n');
M_DAS = compute_das_matrix(t, x_lat, z_ax, x_el, c, Fs, focus);

% Pick sample frames: first, middle, last
sample_idx = unique([1, round(Nframes/2), Nframes]);
fprintf('Sample frames: %s\n', mat2str(sample_idx));

%==========================================================================
% 1) STANDARD B-MODE
%==========================================================================
fprintf('\n=== 1/3: Standard B-mode ===\n');
bmode_dir = fullfile(OUT_FOLDER, 'bmode');
if ~exist(bmode_dir, 'dir'), mkdir(bmode_dir); end

RF_tgc = RF .* reshape(single(TGC), [1, Nt, 1]);
RF_das = permute(RF_tgc, [2 1 3]);
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);
IMG = abs(full(M_DAS * RF_das));
IMG = reshape(IMG, [length(x_lat), length(z_ax), Nframes]);
IMG_max = max(IMG(:));
if IMG_max <= 0, IMG_max = 1; end
IMG_db = 20*log10(IMG / IMG_max);
IMG_disp = permute(IMG_db, [2 1 3]);

dynRange = 60;
for k = sample_idx
    fig = figure('Visible','off');
    imagesc(x_lat*1e3, z_ax*1e3, max(IMG_disp(:,:,k), -dynRange), [-dynRange 0]);
    colormap(gray); axis image; colorbar;
    xlabel('Lateral (mm)'); ylabel('Depth (mm)');
    title(sprintf('B-mode - Frame %d', k));
    exportgraphics(fig, fullfile(bmode_dir, sprintf('bmode_frame_%02d.png', k)), 'Resolution', 150);
    close(fig);
    fprintf('  Saved bmode frame %d\n', k);
end

%==========================================================================
% 2) MICROBUBBLE-ONLY (SVD filtered)
%==========================================================================
fprintf('\n=== 2/3: Microbubble-only (SVD filtered) ===\n');
mb_dir = fullfile(OUT_FOLDER, 'microbubble');
if ~exist(mb_dir, 'dir'), mkdir(mb_dir); end

SVD_CUTOFF = 2;
RF_cas = double(reshape(RF, [Nelem*Nt, Nframes]));
[U, S, V] = svd(RF_cas, 'econ');
S_filt = S;
for i = 1:min(SVD_CUTOFF, size(S,1))
    S_filt(i,i) = 0;
end
RF_svd = single(reshape(U * S_filt * V', [Nelem, Nt, Nframes]));
RF_svd = RF_svd .* reshape(single(TGC), [1, Nt, 1]);

RF_das2 = permute(RF_svd, [2 1 3]);
RF_das2 = hilbert(RF_das2);
RF_das2 = reshape(double(RF_das2), [Nt*Nelem, Nframes]);
IMG2 = abs(full(M_DAS * RF_das2));
IMG2 = reshape(IMG2, [length(x_lat), length(z_ax), Nframes]);
IMG2_max = max(IMG2(:));
if IMG2_max <= 0, IMG2_max = 1; end
IMG2_db = 20*log10(IMG2 / IMG2_max);
IMG2_disp = permute(IMG2_db, [2 1 3]);

dynRange_mb = 30;
for k = sample_idx
    fig = figure('Visible','off');
    imagesc(x_lat*1e3, z_ax*1e3, max(IMG2_disp(:,:,k), -dynRange_mb), [-dynRange_mb 0]);
    colormap(hot); axis image; colorbar;
    xlabel('Lateral (mm)'); ylabel('Depth (mm)');
    title(sprintf('Microbubbles (SVD cutoff=%d) - Frame %d', SVD_CUTOFF, k));
    exportgraphics(fig, fullfile(mb_dir, sprintf('mb_frame_%02d.png', k)), 'Resolution', 150);
    close(fig);
    fprintf('  Saved microbubble frame %d\n', k);
end

%==========================================================================
% 3) B-MODE + GROUND TRUTH OVERLAY
%==========================================================================
fprintf('\n=== 3/3: B-mode + Ground Truth overlay ===\n');
gt_dir = fullfile(OUT_FOLDER, 'bmode_gt');
if ~exist(gt_dir, 'dir'), mkdir(gt_dir); end

% Use SVD-filtered contrast image for GT overlay (same as vis_bmode_groundtruth)
dynRange_gt = 40;

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

for k = sample_idx
    gt_mm = load_gt_frame(GT_FOLDER, k, npad_gt, pulse_name, Geometry);

    fig = figure('Visible','off', 'Color','k', 'InvertHardcopy','off', ...
        'Position', [100 100 500 800]);
    ax = axes(fig, 'Color','k');
    imagesc(ax, x_lat*1e3, z_ax*1e3, max(IMG2_disp(:,:,k), -dynRange_gt), [-dynRange_gt 0]);
    colormap(ax, gray); axis(ax, 'image');
    hold(ax, 'on');
    if ~isempty(gt_mm)
        plot(ax, gt_mm(:,1), gt_mm(:,2), '*', 'Color', [0.3 0.6 1], ...
            'MarkerSize', 6, 'LineWidth', 1);
    end
    hold(ax, 'off');
    xlabel(ax, 'Width [mm]', 'Color','w');
    ylabel(ax, 'Depth [mm]', 'Color','w');
    title(ax, sprintf('Frame %d  (FR=%dHz)', k, round(frameRate)), ...
        'Color','w');
    ax.XColor = 'w'; ax.YColor = 'w';
    exportgraphics(fig, fullfile(gt_dir, sprintf('bmode_gt_frame_%02d.png', k)), ...
        'BackgroundColor','k', 'Resolution', 150);
    close(fig);
    fprintf('  Saved bmode+GT frame %d\n', k);
end

%==========================================================================
% 4) SAMPLE GRID (all frames, B-mode + GT)
%==========================================================================
fprintf('\n=== Bonus: Sample grid ===\n');
Ngrid = min(20, Nframes);
idx_grid = round(linspace(1, Nframes, Ngrid));
nx_g = ceil(sqrt(Ngrid));
ny_g = ceil(Ngrid / nx_g);

fig_grid = figure('Visible','off', 'Color','k', 'InvertHardcopy','off', ...
    'Position', [50 50 1400 800]);
for i = 1:Ngrid
    subplot(ny_g, nx_g, i);
    imagesc(x_lat*1e3, z_ax*1e3, max(IMG2_disp(:,:,idx_grid(i)), -dynRange_gt), [-dynRange_gt 0]);
    colormap(gca, gray); axis image;
    hold on;
    gt_i = load_gt_frame(GT_FOLDER, idx_grid(i), npad_gt, pulse_name, Geometry);
    if ~isempty(gt_i)
        plot(gt_i(:,1), gt_i(:,2), '*', 'Color', [0.3 0.6 1], 'MarkerSize', 3);
    end
    hold off;
    title(sprintf('F%d', idx_grid(i)), 'Color','w');
    set(gca, 'XColor','w', 'YColor','w', 'Color','k');
end
sgtitle(sprintf('B-mode + GT (%d frames)', Nframes), 'Color','w');
exportgraphics(fig_grid, fullfile(OUT_FOLDER, 'sample_grid.png'), ...
    'BackgroundColor','k', 'Resolution', 150);
close(fig_grid);

fprintf('\n=== All visualizations saved to: %s ===\n', OUT_FOLDER);
end

%==========================================================================
% HELPER
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
    gt_mm = [pts(:,2)*1e3, pts(:,1)*1e3];
end
