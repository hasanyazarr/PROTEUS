function process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, ...
    VIZ_OUT, MODE, VIDEO_FPS, PREPROCESSING_OPTIONS)
% PROCESS_RUN  Visualization of a simulated run.
%
% Loads RF data, computes the SVD-filtered + DAS-beamformed B-mode stack, and
% writes the visualization outputs under VIZ_OUT.
%
% Usage:
%   process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, VIZ_OUT)
%   process_run(..., 'preview')              % skip the full-length video
%   process_run(..., 'full', 60)
%
% Args:
%   RESULTS_FOLDER  Folder containing Frame_XXX.mat RF data
%   SETTINGS_PATH   GUI_output_parameters .mat file
%   GT_FOLDER       Ground-truth bubble positions
%   VIZ_OUT         Output dir for bmode_gt/, bmode_clean/, sample_grid.png,
%                   and (in 'full' mode) mb_video.mp4
%   MODE            'full' (default) or 'preview' -- preview skips video
%   VIDEO_FPS       FPS of the per-frame video (default 60). 'full' only.
%   PREPROCESSING_OPTIONS  Preprocessing policy. Optional; defaults to
%                   case_level split with SVD.Cutoff = 2. Fields:
%                   SplitMode: 'case_level' or 'frame_level'.
%                   FitFrameNumbers: source frame numbers used to fit SVD
%                   and intensity normalization. Required for frame_level.
%                   SplitID: label for the preprocessing fit split.
%                   NormalizationMode: 'fit_frames_global_max' or 'per_frame'.
%                   SVD.Cutoff or SVD.Mode = 'adaptive_energy'.
%                   ImageROI: struct with Depth and Lateral, each [min max] in
%                   metres, overriding the vessel-box grid. Optional.

if nargin < 4 || isempty(VIZ_OUT)
    error('process_run:MissingVizOut', 'VIZ_OUT is required -- nothing to do.');
end
if nargin < 5 || isempty(MODE),      MODE = 'full'; end
if nargin < 6 || isempty(VIDEO_FPS), VIDEO_FPS = 60; end
if nargin < 7 || isempty(PREPROCESSING_OPTIONS)
    PREPROCESSING_OPTIONS = struct();
    PREPROCESSING_OPTIONS.SplitMode = 'case_level';
    PREPROCESSING_OPTIONS.SVD.Cutoff = 2;
end

DYNRANGE_VIZ = 60;   % dB, display window for bmode_gt / bmode_clean / video
ROI_MARGIN_LAMBDA = 5;   % wavelengths of margin around the vessel bounding box

%==========================================================================
% LOAD SETTINGS & RF
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
% SVD CLUTTER FILTER
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
% DAS SETUP + TGC
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

[roi_z, roi_x, PreprocessingState.ImageROI] = select_image_roi(...
    PREPROCESSING_OPTIONS, Geometry, lam, ROI_MARGIN_LAMBDA);
x_lat = roi_x(1) : pixelSize : roi_x(2);
z_ax  = roi_z(1) : pixelSize : roi_z(2);
Nx = length(x_lat);
Nz = length(z_ax);
fprintf('Image ROI (%s): depth %.2f-%.2f mm, lateral %.2f-%.2f mm\n', ...
    PreprocessingState.ImageROI.Mode, roi_z*1e3, roi_x*1e3);

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
% BEAMFORM ALL FRAMES
%==========================================================================
fprintf('Beamforming all frames...\n');
RF_das = permute(RF_filt, [2 1 3]);
RF_das = hilbert(RF_das);
RF_das = reshape(double(RF_das), [Nt*Nelem, Nframes]);
IMG = abs(full(M_DAS * RF_das));
IMG = reshape(IMG, [Nx, Nz, Nframes]);
clear RF_filt RF_das M_DAS;

% Fit-scoped normalisation, then dB. The stack is kept unclamped here so the
% display clamp is applied without using held-out frames.
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
% VISUALIZATION
%==========================================================================
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


function [roi_z, roi_x, ROIState] = select_image_roi(...
    options, Geom, lam, margin_lambda)
%SELECT_IMAGE_ROI Axial and lateral extent of the beamformed image, in metres.
%
% The grid used to follow Geometry.Domain, which is sized by the transducer
% surface and the far edge of the medium rather than by the vessel. On the
% 9L-D renal_tree configs that left about 90% of every pixel where no vessel
% can be. The vessel bounding box is already in the settings, so derive the
% grid from it and keep the domain only as a clamp: nothing outside it was
% simulated.
%
% abs(Rotation) turns the box half-extents into the half-extents of its
% axis-aligned envelope, which is exact for the signed permutations the
% configs use and correct for any rotation.

% Rotation is stored as int16 in the settings snapshots. MATLAB refuses to
% multiply an integer matrix by a double one, and normalize_settings_types
% only runs on the copy the notebook writes, so cast here rather than depend
% on it.
D    = Geom.Domain;
half = abs(double(Geom.Rotation)) * (double(Geom.BoundingBox.Diagonal(:)) / 2);
ctr  = double(Geom.Center(:));

if isfield(options, 'ImageROI') && ~isempty(options.ImageROI)
    roi = options.ImageROI;
    if ~isstruct(roi) || ~isfield(roi, 'Depth') || ~isfield(roi, 'Lateral')
        error('process_run:InvalidImageROI', ...
            ['PREPROCESSING_OPTIONS.ImageROI needs Depth and Lateral, ' ...
             'each [min max] in metres.']);
    end
    roi_z = sort(reshape(double(roi.Depth), 1, []));
    roi_x = sort(reshape(double(roi.Lateral), 1, []));
    if numel(roi_z) ~= 2 || numel(roi_x) ~= 2
        error('process_run:InvalidImageROI', ...
            'ImageROI.Depth and ImageROI.Lateral must each hold two values.');
    end
    ROIState.Mode = 'explicit';
    ROIState.MarginWavelengths = [];
else
    margin = margin_lambda * lam;
    roi_z = [ctr(1) - half(1) - margin, ctr(1) + half(1) + margin];
    roi_x = [ctr(2) - half(2) - margin, ctr(2) + half(2) + margin];
    ROIState.Mode = 'vessel_bounding_box';
    ROIState.MarginWavelengths = margin_lambda;
end

roi_z = [max(roi_z(1), 0),              min(roi_z(2), double(D.Xmax))];
roi_x = [max(roi_x(1), double(D.Ymin)), min(roi_x(2), double(D.Ymax))];
if roi_z(2) <= roi_z(1) || roi_x(2) <= roi_x(1)
    error('process_run:EmptyImageROI', ...
        ['Image ROI is empty after clamping to the simulation domain ' ...
         '(depth %.4f-%.4f m, lateral %.4f-%.4f m).'], ...
        roi_z(1), roi_z(2), roi_x(1), roi_x(2));
end

ROIState.VesselBoxDepth   = [ctr(1) - half(1), ctr(1) + half(1)];
ROIState.VesselBoxLateral = [ctr(2) - half(2), ctr(2) + half(2)];
ROIState.DepthRange       = roi_z;
ROIState.LateralRange     = roi_x;
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
