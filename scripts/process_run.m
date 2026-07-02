function process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, ...
    VIZ_OUT, DATASET_OUT, MODE, VIDEO_FPS, SIGMA_PX, ELEVATION_FILTER_MM, ...
    PREPROCESSING_OPTIONS)
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
%   ELEVATION_FILTER_MM  Keep GT labels with |elevation| <= this value
%                   (default 1.0 mm). Pass Inf to disable.
%   PREPROCESSING_OPTIONS  Required for dataset export. Fields:
%                   SplitMode: 'case_level' or 'frame_level'.
%                   FitFrameNumbers: source frame numbers used to fit SVD
%                   and intensity normalization. Required for frame_level.
%                   SplitID: label for the preprocessing fit split.
%                   NormalizationMode: 'fit_frames_global_max' or 'per_frame'.
%                   SVD.Cutoff or SVD.Mode = 'adaptive_energy'.

if nargin < 4, VIZ_OUT     = ''; end
if nargin < 5, DATASET_OUT = ''; end
if nargin < 6 || isempty(MODE),      MODE = 'full'; end
if nargin < 7 || isempty(VIDEO_FPS), VIDEO_FPS = 60; end
if nargin < 8 || isempty(SIGMA_PX),  SIGMA_PX = 1.5; end
if nargin < 9 || isempty(ELEVATION_FILTER_MM), ELEVATION_FILTER_MM = 1.0; end
preprocessing_options_provided = nargin >= 10 && ~isempty(PREPROCESSING_OPTIONS);
if ~preprocessing_options_provided, PREPROCESSING_OPTIONS = struct(); end

do_viz     = ~isempty(VIZ_OUT);
do_dataset = ~isempty(DATASET_OUT);
if ~do_viz && ~do_dataset
    error('process_run: VIZ_OUT and DATASET_OUT both empty -- nothing to do.');
end
if do_dataset && ~preprocessing_options_provided
    error('process_run:MissingPreprocessingOptions', ...
        'Dataset export requires PREPROCESSING_OPTIONS with SplitMode and SVD policy.');
end
if ~do_dataset && ~preprocessing_options_provided
    PREPROCESSING_OPTIONS.SplitMode = 'case_level';
    PREPROCESSING_OPTIONS.SVD.Cutoff = 2;
end

DYNRANGE_VIZ     = 60;   % dB, display window for bmode_gt / bmode_clean / video
DYNRANGE_DATASET = 40;   % dB, clamping range used to normalise LR images to [0,1]
LabelPolicy = build_label_policy(PREPROCESSING_OPTIONS);
VISIBILITY_THRESHOLD = LabelPolicy.VisibilityThreshold; % Minimum local LR value for a label to train on

%==========================================================================
% LOAD SETTINGS & RF  (shared)
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

fprintf('=== process_run: loading RF from %s ===\n', RESULTS_FOLDER);
[RF, sourceFrameNumbers, sourceRFFileNames, pulseInfo] = load_RF_data(...
    RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

[fit_frame_mask, PreprocessingState] = build_preprocessing_state(...
    PREPROCESSING_OPTIONS, sourceFrameNumbers);

%==========================================================================
% SVD CLUTTER FILTER  (shared)
%==========================================================================
RF_fit_cas = double(reshape(RF(:,:,fit_frame_mask), ...
    [Nelem*Nt, sum(fit_frame_mask)]));
[U_fit, S_fit, ~] = svd(RF_fit_cas, 'econ');
singular_values = diag(S_fit);
[n_remove, PreprocessingState.SVD] = select_svd_cutoff(...
    singular_values, PREPROCESSING_OPTIONS);
fprintf('SVD clutter filter (cutoff=%d, mode=%s)...\n', ...
    n_remove, PreprocessingState.SVD.Mode);
RF_cas = double(reshape(RF, [Nelem*Nt, Nframes]));
if n_remove > 0
    clutter_basis = U_fit(:, 1:n_remove);
    RF_cas = RF_cas - clutter_basis * (clutter_basis' * RF_cas);
end
RF_filt = single(reshape(RF_cas, [Nelem, Nt, Nframes]));
PreprocessingState.SVDFitFrameNumbers = sourceFrameNumbers(fit_frame_mask);
PreprocessingState.SVDFitScope = 'specified_source_frames';
clear RF RF_fit_cas RF_cas U_fit S_fit clutter_basis;

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

% Fit-scoped normalisation, then dB. Keep the unclamped stack so each branch
% can clamp at its own dynamic range without using held-out frames.
if strcmpi(PreprocessingState.NormalizationMode, 'per_frame')
    IMG_ref = max(IMG, [], [1,2]);
    IMG_ref(IMG_ref <= 0) = 1;
    IMG_db = 20*log10(IMG ./ IMG_ref);
    PreprocessingState.NormalizationReference = squeeze(IMG_ref);
else
    IMG_ref = max(IMG(:,:,fit_frame_mask), [], 'all');
    if IMG_ref <= 0, IMG_ref = 1; end
    IMG_db = 20*log10(IMG / IMG_ref);
    PreprocessingState.NormalizationReference = IMG_ref;
end
IMG_db = permute(IMG_db, [2 1 3]);            % [Nz, Nx, Nframes]
IMG_db(isnan(IMG_db)) = -Inf;
clear IMG;

%==========================================================================
% Common GT helpers (frame rate, pulse, padding)
%==========================================================================
NFrames_total = Acquisition.NumberOfFrames;
npad_gt    = length(num2str(NFrames_total));
pulse_name = 'Pulse1';
pulseTimes = (pulseInfo.PulseIDsUsed - 1) * Acquisition.TimeBetweenPulses;
pulseInfo.PulseTimes = pulseTimes;

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
        gt_mm = load_gt_frame(GT_FOLDER, sourceFrameNumbers(k), npad_gt, ...
            pulse_name, Geometry);
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
        gt_i = load_gt_frame(GT_FOLDER, sourceFrameNumbers(idx_grid(ig)), ...
            npad_gt, pulse_name, Geometry);
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
                fullfile('mat','gauss_sum'),     fullfile('mat','instance_targets'), ...
                'coordinates'};
    for d = 1:numel(subdirs)
        dpath = fullfile(DATASET_OUT, subdirs{d});
        if ~exist(dpath, 'dir'), mkdir(dpath); end
    end
    fprintf('=== Writing dataset (sigma=%.1f px, |elev|<=%.2f mm) to %s ===\n', ...
        SIGMA_PX, ELEVATION_FILTER_MM, DATASET_OUT);

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
        source_frame = sourceFrameNumbers(iframe);
        lr_frame = LR_all(:,:,iframe);

        [PulseLabels, all_gt_coords_mm, all_gt_coords_px, all_gt_elev_mm, ...
            label_valid, drop_reason, combined_gt_coords_mm, ...
            combined_gt_coords_px, combined_gt_elev_mm, ...
            DroppedLabelCountsByReason, LabelCountsByPulseAndReason] = ...
            load_pulse_labels_for_export(...
                GT_FOLDER, source_frame, npad_gt, pulseInfo, Geometry, ...
                x_lat_mm, z_ax_mm, ELEVATION_FILTER_MM, ...
                VISIBILITY_THRESHOLD, lr_frame);

        gt_mm = combined_gt_coords_mm;
        gt_px = combined_gt_coords_px;
        gt_elev_mm = combined_gt_elev_mm;

        [hr_frame, hr_frame_sum, instance_targets] = render_hr_targets(...
            gt_px, Nz, Nx, SIGMA_PX, kernel_radius);

        gt_coords_mm = gt_mm; gt_coords_px = gt_px;
        sample_metadata.export_index = iframe;
        sample_metadata.source_frame_number = source_frame;
        sample_metadata.source_rf_file = sourceRFFileNames{iframe};
        sample_metadata.PulsingScheme = pulseInfo.PulsingScheme;
        sample_metadata.PulseIDsUsed = pulseInfo.PulseIDsUsed;
        sample_metadata.PulseTimes = pulseInfo.PulseTimes;
        sample_metadata.LabelPulsePolicy = pulseInfo.LabelPulsePolicy;
        sample_metadata.LabelCountsByPulseAndReason = LabelCountsByPulseAndReason;
        save(fullfile(DATASET_OUT, 'mat', 'blob',        ['frame_' frame_tag '.mat']), 'lr_frame', '-v6');
        save(fullfile(DATASET_OUT, 'mat', 'gauss_point', ['frame_' frame_tag '.mat']), 'hr_frame', '-v6');
        save(fullfile(DATASET_OUT, 'mat', 'gauss_sum', ['frame_' frame_tag '.mat']), 'hr_frame_sum', '-v6');
        save(fullfile(DATASET_OUT, 'mat', 'instance_targets', ['frame_' frame_tag '.mat']), ...
            'instance_targets', '-v6');
        save(fullfile(DATASET_OUT, 'coordinates',        ['frame_' frame_tag '.mat']), ...
            'gt_coords_mm', 'gt_coords_px', 'gt_elev_mm', ...
            'all_gt_coords_mm', 'all_gt_coords_px', 'all_gt_elev_mm', ...
            'label_valid', 'drop_reason', 'PulseLabels', ...
            'DroppedLabelCountsByReason', 'LabelCountsByPulseAndReason', ...
            'sample_metadata', '-v6');
        imwrite(uint8(lr_frame * 255), ...
            fullfile(DATASET_OUT, 'frames','blob',        ['frame_' frame_tag '.png']));
        imwrite(uint8(hr_frame * 255), ...
            fullfile(DATASET_OUT, 'frames','gauss_point', ['frame_' frame_tag '.png']));

        if mod(iframe, 50) == 0 || iframe == 1 || iframe == Nframes
            fprintf('  %d / %d  (source frame: %d, valid labels: %d, raw labels: %d)\n', ...
                iframe, Nframes, source_frame, size(gt_px, 1), size(all_gt_coords_px, 1));
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
    metadata.svd_cutoff     = PreprocessingState.SVD.SelectedCutoff;
    metadata.source_frame_numbers = sourceFrameNumbers;
    metadata.source_rf_files = sourceRFFileNames;
    metadata.pulse_info = pulseInfo;
    metadata.preprocessing = PreprocessingState;
    metadata.SplitID = PreprocessingState.SplitID;
    metadata.PreprocessingFitFrames = PreprocessingState.SVDFitFrameNumbers;
    metadata.SVDCutoff = PreprocessingState.SVD.SelectedCutoff;
    metadata.SVDFitScope = PreprocessingState.SVDFitScope;
    metadata.NormalizationMode = PreprocessingState.NormalizationMode;
    metadata.NormalizationReference = PreprocessingState.NormalizationReference;
    metadata.speed_of_sound = c;
    metadata.center_freq    = f0;
    metadata.lambda         = lam;
    metadata.settings_file  = SETTINGS_PATH;
    metadata.results_folder = RESULTS_FOLDER;
    metadata.gt_folder      = GT_FOLDER;
    metadata.elevation_filter_mm = ELEVATION_FILTER_MM;
    metadata.LabelPolicy = LabelPolicy;
    metadata.TargetPolicy.Type = 'legacy_max_plus_sum_and_instance_targets';
    metadata.TargetPolicy.GaussianSigmaPx = SIGMA_PX;
    metadata.TargetPolicy.OverlapComposition = 'max_legacy_sum_density_instance_preserving';
    metadata.TargetPolicy.CoordinateConvention = ...
        'gt_coords_px columns are [lateral_col, axial_row], one-based fractional pixels';
    metadata.DynamicRangeDb = DYNRANGE_DATASET;
    hashes = build_reproducibility_hashes(...
        SETTINGS_PATH, GT_FOLDER, sourceRFFileNames, Geometry);
    metadata.Hashes.SettingsFile = hashes.SettingsFile;
    metadata.Hashes.GTFlowSimulationParameters = ...
        hashes.GTFlowSimulationParameters;
    metadata.Hashes.RFSourceFiles = hashes.RFSourceFiles;
    metadata.Hashes.STLFile = hashes.STLFile;
    metadata.Hashes.VTUFile = hashes.VTUFile;
    metadata.Hashes.GeometryPropertiesFile = hashes.GeometryPropertiesFile;
    metadata.Hashes.STLFilePath = hashes.STLFilePath;
    metadata.Hashes.VTUFilePath = hashes.VTUFilePath;
    metadata.Hashes.GeometryPropertiesFilePath = hashes.GeometryPropertiesFilePath;
    metadata.Pipeline.ExportTimestamp = char(datetime('now', 'TimeZone', 'UTC'));
    [metadata.Pipeline.GitCommit, metadata.Pipeline.GitDirty] = get_git_state();
    metadata.Pipeline.PreprocessingOptions = PREPROCESSING_OPTIONS;
    metadata.Pipeline.PulsingScheme = pulseInfo.PulsingScheme;
    metadata.Pipeline.PulseCombinationFormula = pulseInfo.CombinationFormula;
    metadata.Pipeline.Tiling = get_tiling_metadata(GT_FOLDER);
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


function LabelPolicy = build_label_policy(PREPROCESSING_OPTIONS)
LabelPolicy.VisibilityThreshold = 0.05;
if isfield(PREPROCESSING_OPTIONS, 'LabelPolicy') && ...
        isfield(PREPROCESSING_OPTIONS.LabelPolicy, 'VisibilityThreshold') && ...
        ~isempty(PREPROCESSING_OPTIONS.LabelPolicy.VisibilityThreshold)
    LabelPolicy.VisibilityThreshold = ...
        PREPROCESSING_OPTIONS.LabelPolicy.VisibilityThreshold;
end
LabelPolicy.ValidReasons = {'valid', 'out_of_fov', 'out_of_plane', 'weak_response'};
end


function [fit_frame_mask, PreprocessingState] = build_preprocessing_state(...
    options, sourceFrameNumbers)
if isfield(options, 'SplitMode') && ~isempty(options.SplitMode)
    split_mode = options.SplitMode;
else
    split_mode = 'case_level';
end
if ~any(strcmpi(split_mode, {'case_level', 'frame_level'}))
    error('process_run:InvalidSplitMode', ...
        'SplitMode must be case_level or frame_level.');
end
if strcmpi(split_mode, 'frame_level') && ...
        (~isfield(options, 'FitFrameNumbers') || isempty(options.FitFrameNumbers))
    error('process_run:FrameLevelRequiresFitFrames', ...
        'Frame-level export requires PREPROCESSING_OPTIONS.FitFrameNumbers.');
end
if isfield(options, 'FitFrameNumbers') && ~isempty(options.FitFrameNumbers)
    fit_numbers = reshape(options.FitFrameNumbers, 1, []);
else
    fit_numbers = reshape(sourceFrameNumbers, 1, []);
end
fit_frame_mask = ismember(sourceFrameNumbers, fit_numbers);
if ~any(fit_frame_mask)
    error('process_run:NoPreprocessingFitFrames', ...
        'PREPROCESSING_OPTIONS.FitFrameNumbers does not match loaded RF frames.');
end
if isfield(options, 'SplitID') && ~isempty(options.SplitID)
    split_id = options.SplitID;
else
    split_id = 'all_loaded_frames';
end
if isfield(options, 'NormalizationMode') && ~isempty(options.NormalizationMode)
    norm_mode = options.NormalizationMode;
else
    norm_mode = 'fit_frames_global_max';
end
if ~any(strcmpi(norm_mode, {'fit_frames_global_max', 'per_frame'}))
    error('process_run:InvalidNormalizationMode', ...
        'NormalizationMode must be fit_frames_global_max or per_frame.');
end
PreprocessingState.SplitID = split_id;
PreprocessingState.SplitMode = split_mode;
PreprocessingState.SVDCutoff = [];
PreprocessingState.SVD = struct();
PreprocessingState.SVDFitFrameNumbers = sourceFrameNumbers(fit_frame_mask);
PreprocessingState.SVDFitScope = 'specified_source_frames';
PreprocessingState.NormalizationMode = norm_mode;
PreprocessingState.NormalizationReference = [];
end


function [cutoff, SVDState] = select_svd_cutoff(singular_values, options)
if isfield(options, 'SVD') && isfield(options.SVD, 'Cutoff') && ...
        ~isempty(options.SVD.Cutoff)
    cutoff = options.SVD.Cutoff;
    SVDState.Mode = 'explicit_cutoff';
    SVDState.EnergyThreshold = [];
elseif isfield(options, 'SVD') && isfield(options.SVD, 'Mode') && ...
        strcmpi(options.SVD.Mode, 'adaptive_energy')
    if isfield(options.SVD, 'EnergyThreshold') && ...
            ~isempty(options.SVD.EnergyThreshold)
        threshold = options.SVD.EnergyThreshold;
    else
        threshold = 0.90;
    end
    energy = singular_values.^2;
    cumulative = cumsum(energy) / max(sum(energy), eps);
    cutoff = find(cumulative >= threshold, 1, 'first');
    if isempty(cutoff), cutoff = 0; end
    SVDState.Mode = 'adaptive_energy';
    SVDState.EnergyThreshold = threshold;
else
    error('process_run:MissingSVDPolicy', ...
        'PREPROCESSING_OPTIONS.SVD must specify Cutoff or Mode=adaptive_energy.');
end
cutoff = max(0, min(round(cutoff), numel(singular_values)));
SVDState.SelectedCutoff = cutoff;
SVDState.SingularValues = singular_values;
end


function [PulseLabels, all_gt_coords_mm, all_gt_coords_px, all_gt_elev_mm, ...
    label_valid, drop_reason, combined_gt_coords_mm, combined_gt_coords_px, ...
    combined_gt_elev_mm, DroppedLabelCountsByReason, ...
    LabelCountsByPulseAndReason] = ...
    load_pulse_labels_for_export(gt_folder, frame_idx, npad, pulseInfo, Geom, ...
    x_lat_mm, z_ax_mm, elevation_filter_mm, visibility_threshold, lr_frame)

pulse_ids = pulseInfo.PulseIDsUsed;
PulseLabels = struct('PulseID', {}, 'PulseName', {}, 'gt_coords_mm', {}, ...
    'gt_coords_px', {}, 'gt_elev_mm', {}, 'label_valid', {}, ...
    'drop_reason', {});
all_gt_coords_mm = zeros(0, 2);
all_gt_coords_px = zeros(0, 2);
all_gt_elev_mm = zeros(0, 1);
label_valid = false(0, 1);
drop_reason = {};
LabelCountsByPulseAndReason = struct('PulseID', {}, 'PulseName', {}, ...
    'CountsByReason', {});

for ipulse = 1:numel(pulse_ids)
    pulse_name = ['Pulse' num2str(pulse_ids(ipulse))];
    [pulse_gt_mm, pulse_gt_px, pulse_elev_mm] = load_gt_for_export(...
        gt_folder, frame_idx, npad, pulse_name, Geom, x_lat_mm, z_ax_mm);
    [pulse_valid, pulse_reason] = classify_labels_for_export(...
        pulse_gt_px, pulse_elev_mm, size(lr_frame), elevation_filter_mm, ...
        visibility_threshold, lr_frame);

    PulseLabels(ipulse).PulseID = pulse_ids(ipulse);
    PulseLabels(ipulse).PulseName = pulse_name;
    PulseLabels(ipulse).gt_coords_mm = pulse_gt_mm;
    PulseLabels(ipulse).gt_coords_px = pulse_gt_px;
    PulseLabels(ipulse).gt_elev_mm = pulse_elev_mm;
    PulseLabels(ipulse).label_valid = pulse_valid;
    PulseLabels(ipulse).drop_reason = pulse_reason;
    LabelCountsByPulseAndReason(ipulse).PulseID = pulse_ids(ipulse);
    LabelCountsByPulseAndReason(ipulse).PulseName = pulse_name;
    LabelCountsByPulseAndReason(ipulse).CountsByReason = ...
        count_drop_reasons(pulse_reason);

    all_gt_coords_mm = [all_gt_coords_mm; pulse_gt_mm]; %#ok<AGROW>
    all_gt_coords_px = [all_gt_coords_px; pulse_gt_px]; %#ok<AGROW>
    all_gt_elev_mm = [all_gt_elev_mm; pulse_elev_mm]; %#ok<AGROW>
    label_valid = [label_valid; pulse_valid]; %#ok<AGROW>
    drop_reason = [drop_reason; pulse_reason(:)]; %#ok<AGROW>
end

combined_gt_coords_mm = all_gt_coords_mm(label_valid, :);
combined_gt_coords_px = all_gt_coords_px(label_valid, :);
combined_gt_elev_mm = all_gt_elev_mm(label_valid);
DroppedLabelCountsByReason = count_drop_reasons(drop_reason);
end


function [valid, drop_reason] = classify_labels_for_export(...
    gt_px, elev_mm, image_size, elevation_filter_mm, visibility_threshold, lr_frame)
n = size(gt_px, 1);
valid = true(n, 1);
drop_reason = repmat({'valid'}, n, 1);
for i = 1:n
    col = gt_px(i, 1);
    row = gt_px(i, 2);
    if isnan(col) || isnan(row)
        valid(i) = false;
        drop_reason{i} = 'out_of_fov';
    elseif row < 1 || row > image_size(1) || col < 1 || col > image_size(2)
        valid(i) = false;
        drop_reason{i} = 'out_of_fov';
    elseif abs(elev_mm(i)) > elevation_filter_mm
        valid(i) = false;
        drop_reason{i} = 'out_of_plane';
    else
        r0 = max(1, floor(row) - 1);
        r1 = min(image_size(1), floor(row) + 1);
        c0 = max(1, floor(col) - 1);
        c1 = min(image_size(2), floor(col) + 1);
        local_peak = max(lr_frame(r0:r1, c0:c1), [], 'all');
        if local_peak < visibility_threshold
            valid(i) = false;
            drop_reason{i} = 'weak_response';
        end
    end
end
end


function counts = count_drop_reasons(drop_reason)
counts.valid = sum(strcmp(drop_reason, 'valid'));
counts.out_of_fov = sum(strcmp(drop_reason, 'out_of_fov'));
counts.out_of_plane = sum(strcmp(drop_reason, 'out_of_plane'));
counts.weak_response = sum(strcmp(drop_reason, 'weak_response'));
end


function [hr_frame, hr_frame_sum, instance_targets] = render_hr_targets(...
    gt_px, Nz, Nx, sigma_px, kernel_radius)
hr_frame = zeros(Nz, Nx, 'single');      % Legacy max-composed heatmap
hr_frame_sum = zeros(Nz, Nx, 'single');  % Density map preserving overlap mass
instance_targets = struct('center_px', {}, 'rows', {}, 'cols', {}, 'values', {});
for b = 1:size(gt_px, 1)
    col_center = gt_px(b, 1);
    row_center = gt_px(b, 2);
    r_min = max(1,  floor(row_center) - kernel_radius);
    r_max = min(Nz, floor(row_center) + kernel_radius);
    c_min = max(1,  floor(col_center) - kernel_radius);
    c_max = min(Nx, floor(col_center) + kernel_radius);
    rows = [];
    cols = [];
    values = [];
    for r = r_min:r_max
        for cc = c_min:c_max
            d2 = (cc - col_center)^2 + (r - row_center)^2;
            val = single(exp(-d2 / (2 * sigma_px^2)));
            if val > hr_frame(r, cc)
                hr_frame(r, cc) = val;
            end
            hr_frame_sum(r, cc) = hr_frame_sum(r, cc) + val;
            rows(end+1, 1) = r; %#ok<AGROW>
            cols(end+1, 1) = cc; %#ok<AGROW>
            values(end+1, 1) = val; %#ok<AGROW>
        end
    end
    instance_targets(b).center_px = [col_center, row_center]; %#ok<AGROW>
    instance_targets(b).rows = rows; %#ok<AGROW>
    instance_targets(b).cols = cols; %#ok<AGROW>
    instance_targets(b).values = values; %#ok<AGROW>
end
hr_frame_sum = min(hr_frame_sum, 1);
end


function hashes = build_reproducibility_hashes(...
    settings_path, gt_folder, source_rf_files, Geometry)
hashes.SettingsFile = file_hash(settings_path);
hashes.GTFlowSimulationParameters = file_hash(...
    fullfile(gt_folder, 'FlowSimulationParameters.mat'));
hashes.RFSourceFiles = cell(size(source_rf_files));
for i = 1:numel(source_rf_files)
    hashes.RFSourceFiles{i} = file_hash(source_rf_files{i});
end
[stl_file, vtu_file, geometry_properties_file] = geometry_source_files(Geometry);
hashes.STLFile = file_hash(stl_file);
hashes.VTUFile = file_hash(vtu_file);
hashes.GeometryPropertiesFile = file_hash(geometry_properties_file);
hashes.STLFilePath = stl_file;
hashes.VTUFilePath = vtu_file;
hashes.GeometryPropertiesFilePath = geometry_properties_file;
end


function [stl_file, vtu_file, geometry_properties_file] = geometry_source_files(Geometry)
stl_file = '';
vtu_file = '';
geometry_properties_file = '';
if isfield(Geometry, 'GeometriesPath') && isfield(Geometry, 'Folder')
    geometry_folder = fullfile(Geometry.GeometriesPath, Geometry.Folder);
    vtu_file = fullfile(geometry_folder, 'vtu.mat');
    geometry_properties_file = fullfile(geometry_folder, 'GeometryProperties.mat');
    if isfield(Geometry, 'STLfile')
        stl_file = fullfile(geometry_folder, Geometry.STLfile);
    end
end
end


function [git_commit, git_dirty] = get_git_state()
[status_commit, out_commit] = system('git rev-parse HEAD');
if status_commit == 0
    git_commit = strtrim(out_commit);
else
    git_commit = '';
end
[status_dirty, out_dirty] = system('git status --short');
git_dirty = status_dirty == 0 && ~isempty(strtrim(out_dirty));
end


function tiling = get_tiling_metadata(gt_folder)
tiling = struct();
fsp_file = fullfile(gt_folder, 'FlowSimulationParameters.mat');
if exist(fsp_file, 'file')
    data = load(fsp_file, 'FlowSimulationParameters');
    if isfield(data.FlowSimulationParameters, 'Tiling')
        tiling = data.FlowSimulationParameters.Tiling;
    end
end
end


function [gt_mm, gt_px, gt_elev_mm] = load_gt_for_export(gt_folder, frame_idx, npad, ...
    pulse_name, Geom, x_lat_mm, z_ax_mm)
gt_file = fullfile(gt_folder, sprintf('Frame_%s.mat', ...
    num2str(frame_idx, ['%0' num2str(npad) 'd'])));
gt_mm = zeros(0, 2); gt_px = zeros(0, 2); gt_elev_mm = zeros(0, 1);
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
elev_mm = pts(:,3) * 1e3;
gt_mm  = [lat_mm, ax_mm];
col_px = interp1(x_lat_mm, 1:length(x_lat_mm), lat_mm, 'linear', NaN);
row_px = interp1(z_ax_mm,  1:length(z_ax_mm),  ax_mm,  'linear', NaN);
gt_px = [col_px, row_px];
gt_elev_mm = elev_mm;
end
