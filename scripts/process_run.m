function process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, ...
    VIZ_OUT, DATASET_OUT, MODE, VIDEO_FPS, SIGMA_PX)
% PROCESS_RUN  Combined visualization + super-resolution dataset export.
%
% Loads RF data, computes the SVD-filtered + DAS-beamformed B-mode stack
% ONCE, and then writes both the visualization outputs (under VIZ_OUT) and
% the LR/HR training pairs (under DATASET_OUT). Pass '' for either output
% path to skip that branch.
%
% This replaces back-to-back calls to visualize_all + dataset_export, which
% otherwise duplicated the entire DAS pipeline.
%
% Usage:
%   process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, VIZ_OUT, DATASET_OUT)
%   process_run(..., 'preview')              % skip the full-length video
%   process_run(..., 'full', 60, 1.5)
%
% Args:
%   RESULTS_FOLDER  Folder containing Frame_XXX.mat RF data
%   SETTINGS_PATH   GUI_output_parameters .mat file
%   GT_FOLDER       Ground-truth bubble positions
%   VIZ_OUT         Output dir for bmode_gt/, bmode_clean/, sample_grid.png,
%                   and (in 'full' mode) mb_video.mp4
%   DATASET_OUT     Output dir for frames/, mat/, coordinates/, metadata.mat
%   MODE            'full' (default) or 'preview' -- preview skips video
%   VIDEO_FPS       FPS of the per-frame video (default 60). 'full' only.
%   SIGMA_PX        Gaussian sigma (px) for the HR target (default 1.5)

if nargin < 4, VIZ_OUT     = ''; end
if nargin < 5, DATASET_OUT = ''; end
if nargin < 6 || isempty(MODE),      MODE = 'full'; end
if nargin < 7 || isempty(VIDEO_FPS), VIDEO_FPS = 60; end
if nargin < 8 || isempty(SIGMA_PX),  SIGMA_PX = 1.5; end

do_viz     = ~isempty(VIZ_OUT);
do_dataset = ~isempty(DATASET_OUT);
if ~do_viz && ~do_dataset
    error('process_run: VIZ_OUT and DATASET_OUT both empty -- nothing to do.');
end

SVD_CUTOFF       = 2;
DYNRANGE_VIZ     = 60;   % dB, display window for bmode_gt / bmode_clean / video
DYNRANGE_DATASET = 40;   % dB, clamping range used to normalise LR images to [0,1]

%==========================================================================
% LOAD SETTINGS & RF  (shared)
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

fprintf('=== process_run: loading RF from %s ===\n', RESULTS_FOLDER);
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% SVD CLUTTER FILTER  (shared)
%==========================================================================
fprintf('SVD clutter filter (cutoff=%d)...\n', SVD_CUTOFF);
RF_cas = double(reshape(RF, [Nelem*Nt, Nframes]));
[U, S, V] = svd(RF_cas, 'econ');
for i = 1:min(SVD_CUTOFF, size(S,1))
    S(i,i) = 0;
end
RF_filt = single(reshape(U * S * V', [Nelem, Nt, Nframes]));
clear RF RF_cas U S V;

%==========================================================================
% DAS SETUP + TGC  (shared)
%==========================================================================
Fs = SimulationParameters.SamplingRate;
t  = (0:(Nt-1)) / Fs;

p     = Transducer.Pitch;
x_el  = -p/2*(Nelem-1) + (0:(Nelem-1))*p;
focus = Transmit.LateralFocus;

c   = Medium.SpeedOfSound;
f0  = Transmit.CenterFrequency;
lam = c / f0;
pixelSize = lam / 5;

D = Geometry.Domain;
width = D.Ymax - D.Ymin;
depth = D.Xmax;
x_lat = -width/2 : pixelSize : width/2;
z_ax  = 0        : pixelSize : depth;
Nx = length(x_lat);
Nz = length(z_ax);

% Time corrections
IR     = Transducer.ReceiveImpulseResponse;
V_ref  = conv(Transmit.PressureSignal, IR) / Transducer.SamplingRate;
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
RF_filt = RF_filt .* reshape(single(TGC), [1, Nt, 1]);

% DAS matrix
fprintf('Computing DAS matrix (%d x %d pixels)...\n', Nz, Nx);
M_DAS = compute_das_matrix(t, x_lat, z_ax, x_el, c, Fs, focus);

%==========================================================================
% BEAMFORM ALL FRAMES  (shared)
%==========================================================================
fprintf('Beamforming all frames...\n');
RF_das = permute(RF_filt, [2 1 3]);
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);
IMG = abs(full(M_DAS * RF_das));
IMG = reshape(IMG, [Nx, Nz, Nframes]);
clear RF_filt RF_das M_DAS;

% Single global normalisation, then dB. Keep the unclamped stack so each
% branch can clamp at its own dynamic range.
IMG_max = max(IMG(:));
if IMG_max <= 0, IMG_max = 1; end
IMG_db = 20*log10(IMG / IMG_max);
IMG_db = permute(IMG_db, [2 1 3]);            % [Nz, Nx, Nframes]
IMG_db(isnan(IMG_db)) = -Inf;
clear IMG;

%==========================================================================
% Common GT helpers (frame rate, pulse, padding)
%==========================================================================
NFrames_total = Acquisition.NumberOfFrames;
npad_gt    = length(num2str(NFrames_total));
pulse_name = 'Pulse1';

fsp_file = fullfile(GT_FOLDER, 'FlowSimulationParameters.mat');
if exist(fsp_file, 'file')
    fsp = load(fsp_file, 'FlowSimulationParameters');
    frameRate = fsp.FlowSimulationParameters.FrameRate;
else
    frameRate = Acquisition.FrameRate;
end

%==========================================================================
% VISUALIZATION BRANCH
%==========================================================================
if do_viz
    if ~exist(VIZ_OUT, 'dir'), mkdir(VIZ_OUT); end
    fprintf('=== Writing viz outputs to %s ===\n', VIZ_OUT);

    IMG_disp = max(IMG_db, -DYNRANGE_VIZ);

    sample_idx = unique([1, round(Nframes/2), Nframes]);

    % --- bmode_gt ---
    gt_dir = fullfile(VIZ_OUT, 'bmode_gt');
    if ~exist(gt_dir, 'dir'), mkdir(gt_dir); end
    for k = sample_idx
        gt_mm = load_gt_frame(GT_FOLDER, k, npad_gt, pulse_name, Geometry);
        fig = figure('Visible','off', 'Color','k', 'InvertHardcopy','off', ...
            'Position', [100 100 500 800]);
        ax = axes(fig, 'Color','k');
        imagesc(ax, x_lat*1e3, z_ax*1e3, IMG_disp(:,:,k), [-DYNRANGE_VIZ 0]);
        colormap(ax, gray); axis(ax, 'image');
        hold(ax, 'on');
        if ~isempty(gt_mm)
            plot(ax, gt_mm(:,1), gt_mm(:,2), '*', 'Color', [0.3 0.6 1], ...
                'MarkerSize', 6, 'LineWidth', 1);
        end
        hold(ax, 'off');
        xlabel(ax, 'Width [mm]', 'Color','w');
        ylabel(ax, 'Depth [mm]', 'Color','w');
        title(ax, sprintf('Frame %d  (FR=%dHz)', k, round(frameRate)), 'Color','w');
        ax.XColor = 'w'; ax.YColor = 'w';
        exportgraphics(fig, fullfile(gt_dir, sprintf('bmode_gt_frame_%02d.png', k)), ...
            'BackgroundColor','k', 'Resolution', 150);
        close(fig);
    end
    fprintf('  bmode_gt: %d sample frames\n', numel(sample_idx));

    % --- bmode_clean ---
    clean_dir = fullfile(VIZ_OUT, 'bmode_clean');
    if ~exist(clean_dir, 'dir'), mkdir(clean_dir); end
    for k = sample_idx
        fig = figure('Visible','off', 'Color','k', 'InvertHardcopy','off');
        ax = axes(fig, 'Position', [0 0 1 1]);
        imagesc(ax, IMG_disp(:,:,k), [-DYNRANGE_VIZ 0]);
        colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');
        exportgraphics(fig, fullfile(clean_dir, sprintf('bmode_clean_frame_%02d.png', k)), ...
            'BackgroundColor','k', 'Resolution', 150);
        close(fig);
    end
    fprintf('  bmode_clean: %d sample frames\n', numel(sample_idx));

    % --- sample grid ---
    Ngrid = min(20, Nframes);
    idx_grid = round(linspace(1, Nframes, Ngrid));
    nx_g = ceil(sqrt(Ngrid)); ny_g = ceil(Ngrid / nx_g);
    fig_g = figure('Visible','off', 'Color','k', 'InvertHardcopy','off', ...
        'Position', [50 50 1400 800]);
    for ig = 1:Ngrid
        subplot(ny_g, nx_g, ig);
        imagesc(x_lat*1e3, z_ax*1e3, IMG_disp(:,:,idx_grid(ig)), [-DYNRANGE_VIZ 0]);
        colormap(gca, gray); axis image;
        hold on;
        gt_i = load_gt_frame(GT_FOLDER, idx_grid(ig), npad_gt, pulse_name, Geometry);
        if ~isempty(gt_i)
            plot(gt_i(:,1), gt_i(:,2), '*', 'Color', [0.3 0.6 1], 'MarkerSize', 3);
        end
        hold off;
        title(sprintf('F%d', idx_grid(ig)), 'Color','w');
        set(gca, 'XColor','w', 'YColor','w', 'Color','k');
    end
    sgtitle(sprintf('B-mode + GT (%d frames)', Nframes), 'Color','w');
    exportgraphics(fig_g, fullfile(VIZ_OUT, 'sample_grid.png'), ...
        'BackgroundColor','k', 'Resolution', 150);
    close(fig_g);
    fprintf('  sample_grid: %d frames\n', Ngrid);

    % --- full-length video (skipped in preview mode) ---
    if strcmpi(MODE, 'full')
        write_full_video(VIZ_OUT, IMG_disp, x_lat, z_ax, frameRate, ...
            DYNRANGE_VIZ, VIDEO_FPS);
    else
        fprintf('  preview mode: skipping mb_video (use MODE=''full'' to render)\n');
    end
end

%==========================================================================
% DATASET BRANCH
%==========================================================================
if do_dataset
    if ~exist(DATASET_OUT, 'dir'), mkdir(DATASET_OUT); end
    subdirs = { fullfile('frames','blob'),       fullfile('frames','gauss_point'), ...
                fullfile('mat','blob'),          fullfile('mat','gauss_point'), ...
                'coordinates'};
    for d = 1:numel(subdirs)
        dpath = fullfile(DATASET_OUT, subdirs{d});
        if ~exist(dpath, 'dir'), mkdir(dpath); end
    end
    fprintf('=== Writing dataset (sigma=%.1f px) to %s ===\n', SIGMA_PX, DATASET_OUT);

    % LR images: clamp at DYNRANGE_DATASET, normalise to [0,1]
    IMG_db_ds = max(IMG_db, -DYNRANGE_DATASET);
    LR_all = single((IMG_db_ds + DYNRANGE_DATASET) / DYNRANGE_DATASET);
    clear IMG_db_ds;

    npad     = length(num2str(Nframes));
    x_lat_mm = x_lat * 1e3;
    z_ax_mm  = z_ax  * 1e3;
    kernel_radius = ceil(4 * SIGMA_PX);

    for iframe = 1:Nframes
        frame_tag = num2str(iframe, ['%0' num2str(npad) 'd']);
        lr_frame = LR_all(:,:,iframe);

        [gt_mm, gt_px] = load_gt_for_export(GT_FOLDER, iframe, npad_gt, ...
            pulse_name, Geometry, x_lat_mm, z_ax_mm);

        hr_frame = zeros(Nz, Nx, 'single');
        for b = 1:size(gt_px, 1)
            col_center = gt_px(b, 1);
            row_center = gt_px(b, 2);
            r_min = max(1,  floor(row_center) - kernel_radius);
            r_max = min(Nz, floor(row_center) + kernel_radius);
            c_min = max(1,  floor(col_center) - kernel_radius);
            c_max = min(Nx, floor(col_center) + kernel_radius);
            for r = r_min:r_max
                for cc = c_min:c_max
                    d2 = (cc - col_center)^2 + (r - row_center)^2;
                    val = exp(-d2 / (2 * SIGMA_PX^2));
                    if val > hr_frame(r, cc)
                        hr_frame(r, cc) = val;
                    end
                end
            end
        end

        gt_coords_mm = gt_mm; gt_coords_px = gt_px;
        save(fullfile(DATASET_OUT, 'mat', 'blob',        ['frame_' frame_tag '.mat']), 'lr_frame', '-v6');
        save(fullfile(DATASET_OUT, 'mat', 'gauss_point', ['frame_' frame_tag '.mat']), 'hr_frame', '-v6');
        save(fullfile(DATASET_OUT, 'coordinates',        ['frame_' frame_tag '.mat']), ...
            'gt_coords_mm', 'gt_coords_px', '-v6');
        imwrite(uint8(lr_frame * 255), ...
            fullfile(DATASET_OUT, 'frames','blob',        ['frame_' frame_tag '.png']));
        imwrite(uint8(hr_frame * 255), ...
            fullfile(DATASET_OUT, 'frames','gauss_point', ['frame_' frame_tag '.png']));

        if mod(iframe, 50) == 0 || iframe == 1 || iframe == Nframes
            fprintf('  %d / %d  (bubbles in frame: %d)\n', iframe, Nframes, size(gt_px, 1));
        end
    end

    metadata.x_lat_mm       = x_lat_mm;
    metadata.z_ax_mm        = z_ax_mm;
    metadata.pixel_size_m   = pixelSize;
    metadata.pixel_size_mm  = pixelSize * 1e3;
    metadata.image_size     = [Nz, Nx];
    metadata.num_frames     = Nframes;
    metadata.sigma_px       = SIGMA_PX;
    metadata.dynamic_range  = DYNRANGE_DATASET;
    metadata.svd_cutoff     = SVD_CUTOFF;
    metadata.speed_of_sound = c;
    metadata.center_freq    = f0;
    metadata.lambda         = lam;
    metadata.settings_file  = SETTINGS_PATH;
    metadata.results_folder = RESULTS_FOLDER;
    metadata.gt_folder      = GT_FOLDER;
    save(fullfile(DATASET_OUT, 'metadata.mat'), 'metadata', '-v6');
    fprintf('  metadata.mat saved\n');
end

fprintf('\n=== process_run complete ===\n');

end


%==========================================================================
% Helpers
%==========================================================================

function write_full_video(VIZ_OUT, IMG_disp, x_lat, z_ax, frameRate, ...
                          DYNRANGE_VIZ, VIDEO_FPS)

Nframes_v = size(IMG_disp, 3);
mp4_path = fullfile(VIZ_OUT, 'mb_video.mp4');
avi_path = fullfile(VIZ_OUT, 'mb_video.avi');
use_mp4 = true;
try
    vw = VideoWriter(mp4_path, 'MPEG-4');
    vw.FrameRate = VIDEO_FPS;
    vw.Quality   = 95;
    open(vw);
catch
    fprintf('  MPEG-4 unavailable, falling back to Motion JPEG AVI...\n');
    use_mp4 = false;
    vw = VideoWriter(avi_path, 'Motion JPEG AVI');
    vw.FrameRate = VIDEO_FPS;
    vw.Quality   = 95;
    open(vw);
end

fig_v = figure('Visible','off', 'Color','k', 'InvertHardcopy','off', ...
    'Position', [100 100 800 1200]);
ax_v = axes(fig_v, 'Color','k');
for k = 1:Nframes_v
    imagesc(ax_v, x_lat*1e3, z_ax*1e3, IMG_disp(:,:,k), [-DYNRANGE_VIZ 0]);
    colormap(ax_v, gray); axis(ax_v, 'image');
    xlabel(ax_v, 'Width [mm]', 'Color','w');
    ylabel(ax_v, 'Depth [mm]', 'Color','w');
    title(ax_v, sprintf('Frame %d/%d  (FR=%dHz)', k, Nframes_v, round(frameRate)), 'Color','w');
    ax_v.XColor = 'w'; ax_v.YColor = 'w';
    drawnow;
    writeVideo(vw, getframe(fig_v).cdata);
    if mod(k, 100) == 0 || k == Nframes_v
        fprintf('  video: %d/%d frames\n', k, Nframes_v);
    end
end
close(vw); close(fig_v);
if use_mp4
    fprintf('  saved video: %s\n', mp4_path);
else
    fprintf('  saved video: %s\n', avi_path);
end

end


function gt_mm = load_gt_frame(gt_folder, frame_idx, npad, pulse_name, Geom)
gt_file = fullfile(gt_folder, sprintf('Frame_%s.mat', ...
    num2str(frame_idx, ['%0' num2str(npad) 'd'])));
gt_mm = zeros(0, 2);
if ~exist(gt_file, 'file'), return; end
gt_data = load(gt_file, 'Frame');
if ~isfield(gt_data.Frame, pulse_name), return; end
pts = gt_data.Frame.(pulse_name).Points';
pts = pts - Geom.BoundingBox.Center;
pts = Geom.Rotation * pts;
pts = pts + Geom.Center;
pts = pts';
gt_mm = [pts(:,2)*1e3, pts(:,1)*1e3];
end


function [gt_mm, gt_px] = load_gt_for_export(gt_folder, frame_idx, npad, ...
    pulse_name, Geom, x_lat_mm, z_ax_mm)
gt_file = fullfile(gt_folder, sprintf('Frame_%s.mat', ...
    num2str(frame_idx, ['%0' num2str(npad) 'd'])));
gt_mm = zeros(0, 2); gt_px = zeros(0, 2);
if ~exist(gt_file, 'file'), return; end
gt_data = load(gt_file, 'Frame');
if ~isfield(gt_data.Frame, pulse_name), return; end
pts = gt_data.Frame.(pulse_name).Points';
pts = pts - Geom.BoundingBox.Center;
pts = Geom.Rotation * pts;
pts = pts + Geom.Center;
pts = pts';
lat_mm = pts(:,2) * 1e3;
ax_mm  = pts(:,1) * 1e3;
gt_mm  = [lat_mm, ax_mm];
col_px = interp1(x_lat_mm, 1:length(x_lat_mm), lat_mm, 'linear', NaN);
row_px = interp1(z_ax_mm,  1:length(z_ax_mm),  ax_mm,  'linear', NaN);
valid  = ~isnan(col_px) & ~isnan(row_px);
gt_mm = gt_mm(valid, :);
gt_px = [col_px(valid), row_px(valid)];
end
