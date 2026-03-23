function dataset_export(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER, SIGMA_PX)
% DATASET_EXPORT  Export paired LR/HR data for super-resolution training.
%
% Outputs (saved to OUT_FOLDER):
%   frames/blob/          - LR B-mode PNG images (SVD-filtered, beamformed)
%   frames/gauss_point/   - HR Gaussian-rendered GT bubble maps as PNG
%   mat/blob/             - LR B-mode as .mat (float32, 0-1)
%   mat/gauss_point/      - HR Gaussian maps as .mat (float32, 0-1)
%   coordinates/          - Raw GT bubble coordinates per frame (.mat)
%   metadata.mat          - Pixel grid, physical dimensions, parameters
%
% Usage:
%   dataset_export(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER)
%   dataset_export(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER, 1.5)
%
% Arguments:
%   RESULTS_FOLDER - Path to RF data (Frame_XXX.mat files)
%   SETTINGS_PATH  - Path to GUI_output_parameters .mat file
%   GT_FOLDER      - Path to ground truth frames
%   OUT_FOLDER     - Output directory for the dataset
%   SIGMA_PX       - Gaussian sigma in pixels for HR target (default: 1.5)

if nargin < 5 || isempty(SIGMA_PX)
    SIGMA_PX = 1.5;
end

SVD_CUTOFF = 2;
DYNAMIC_RANGE = 40; % dB

% Create output directories
subdirs = { ...
    fullfile('frames', 'blob'), ...
    fullfile('frames', 'gauss_point'), ...
    fullfile('mat',    'blob'), ...
    fullfile('mat',    'gauss_point'), ...
    'coordinates'};
for d = 1:length(subdirs)
    dpath = fullfile(OUT_FOLDER, subdirs{d});
    if ~exist(dpath, 'dir'), mkdir(dpath); end
end

%==========================================================================
% LOAD SETTINGS & RF DATA
%==========================================================================
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', ...
    'SimulationParameters', 'Transducer', 'Transmit');

fprintf('=== Dataset Export ===\n');
fprintf('Loading RF data from %s\n', RESULTS_FOLDER);
RF = load_RF_data(RESULTS_FOLDER, Acquisition.PulsingScheme);
[Nelem, Nt, Nframes] = size(RF);
fprintf('  %d elements, %d samples, %d frames\n', Nelem, Nt, Nframes);

%==========================================================================
% SVD CLUTTER FILTER
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
% DAS BEAMFORMING SETUP
%==========================================================================
fprintf('Setting up DAS beamforming...\n');
Fs = SimulationParameters.SamplingRate;
t = (0:(Nt-1)) / Fs;

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
x_lat = -width/2 : pixelSize : width/2;   % lateral axis [m]
z_ax  = 0 : pixelSize : depth;            % axial axis [m]

Nx = length(x_lat);
Nz = length(z_ax);

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

% Log compression
IMG_max = max(IMG(:));
if IMG_max <= 0, IMG_max = 1; end
IMG_db = 20*log10(IMG / IMG_max);
clear IMG;

% Transpose to [Nz x Nx x Nframes] (rows=axial, cols=lateral)
IMG_db = permute(IMG_db, [2 1 3]);
IMG_db(isnan(IMG_db)) = -DYNAMIC_RANGE;
IMG_db(IMG_db < -DYNAMIC_RANGE) = -DYNAMIC_RANGE;

% Normalize to [0, 1]
LR_all = (IMG_db + DYNAMIC_RANGE) / DYNAMIC_RANGE;
clear IMG_db;

%==========================================================================
% GT SETUP
%==========================================================================
NFrames_total = Acquisition.NumberOfFrames;
npad_gt = length(num2str(NFrames_total));
pulse_name = 'Pulse1';

% Gaussian stamp radius
kernel_radius = ceil(4 * SIGMA_PX);

%==========================================================================
% EXPORT FRAME BY FRAME
%==========================================================================
fprintf('Exporting %d frames (sigma=%.1f px)...\n', Nframes, SIGMA_PX);
npad = length(num2str(Nframes));

x_lat_mm = x_lat * 1e3;
z_ax_mm  = z_ax  * 1e3;

for iframe = 1:Nframes
    frame_tag = num2str(iframe, ['%0' num2str(npad) 'd']);

    %--- LR (blob) frame ---
    lr_frame = single(LR_all(:,:,iframe));

    %--- GT coordinates ---
    [gt_mm, gt_px] = load_gt_for_export(GT_FOLDER, iframe, npad_gt, ...
        pulse_name, Geometry, x_lat_mm, z_ax_mm);

    %--- HR (gauss_point) frame: render Gaussians ---
    hr_frame = zeros(Nz, Nx, 'single');
    for b = 1:size(gt_px, 1)
        col_center = gt_px(b, 1);
        row_center = gt_px(b, 2);

        r_min = max(1, floor(row_center) - kernel_radius);
        r_max = min(Nz, floor(row_center) + kernel_radius);
        c_min = max(1, floor(col_center) - kernel_radius);
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

    %--- Save .mat ---
    gt_coords_mm = gt_mm;
    gt_coords_px = gt_px;

    save(fullfile(OUT_FOLDER, 'mat', 'blob',        ['frame_' frame_tag '.mat']), 'lr_frame', '-v6');
    save(fullfile(OUT_FOLDER, 'mat', 'gauss_point',  ['frame_' frame_tag '.mat']), 'hr_frame', '-v6');
    save(fullfile(OUT_FOLDER, 'coordinates',         ['frame_' frame_tag '.mat']), ...
        'gt_coords_mm', 'gt_coords_px', '-v6');

    %--- Save .png ---
    imwrite(uint8(lr_frame * 255), ...
        fullfile(OUT_FOLDER, 'frames', 'blob',        ['frame_' frame_tag '.png']));
    imwrite(uint8(hr_frame * 255), ...
        fullfile(OUT_FOLDER, 'frames', 'gauss_point',  ['frame_' frame_tag '.png']));

    if mod(iframe, 50) == 0 || iframe == 1 || iframe == Nframes
        fprintf('  %d / %d  (bubbles in frame: %d)\n', iframe, Nframes, size(gt_px,1));
    end
end

%==========================================================================
% SAVE METADATA
%==========================================================================
metadata.x_lat_mm      = x_lat_mm;
metadata.z_ax_mm       = z_ax_mm;
metadata.pixel_size_m  = pixelSize;
metadata.pixel_size_mm = pixelSize * 1e3;
metadata.image_size    = [Nz, Nx];
metadata.num_frames    = Nframes;
metadata.sigma_px      = SIGMA_PX;
metadata.dynamic_range = DYNAMIC_RANGE;
metadata.svd_cutoff    = SVD_CUTOFF;
metadata.speed_of_sound = c;
metadata.center_freq    = f0;
metadata.lambda         = lambda;
metadata.settings_file  = SETTINGS_PATH;
metadata.results_folder = RESULTS_FOLDER;
metadata.gt_folder      = GT_FOLDER;

save(fullfile(OUT_FOLDER, 'metadata.mat'), 'metadata', '-v6');

fprintf('\n=== Dataset export complete ===\n');
fprintf('  Output:     %s\n', OUT_FOLDER);
fprintf('  Frames:     %d\n', Nframes);
fprintf('  Image size: %d x %d (axial x lateral)\n', Nz, Nx);
fprintf('  Pixel size: %.4f mm\n', pixelSize*1e3);
fprintf('  Sigma:      %.1f px\n', SIGMA_PX);
end

%==========================================================================
% HELPER: Load GT and convert to both mm and pixel coordinates
%==========================================================================
function [gt_mm, gt_px] = load_gt_for_export(gt_folder, frame_idx, npad, ...
    pulse_name, Geom, x_lat_mm, z_ax_mm)

    gt_file = fullfile(gt_folder, sprintf('Frame_%s.mat', ...
        num2str(frame_idx, ['%0' num2str(npad) 'd'])));

    gt_mm = zeros(0, 2);
    gt_px = zeros(0, 2);

    if ~exist(gt_file, 'file'), return; end
    gt_data = load(gt_file, 'Frame');
    if ~isfield(gt_data.Frame, pulse_name), return; end

    pts_raw = gt_data.Frame.(pulse_name).Points;
    pts = pts_raw';
    pts = pts - Geom.BoundingBox.Center;
    pts = Geom.Rotation * pts;
    pts = pts + Geom.Center;
    pts = pts';

    % Physical coordinates in mm: [lateral, axial]
    lat_mm = pts(:,2) * 1e3;
    ax_mm  = pts(:,1) * 1e3;
    gt_mm  = [lat_mm, ax_mm];

    % Convert mm to pixel coordinates (fractional)
    col_px = interp1(x_lat_mm, 1:length(x_lat_mm), lat_mm, 'linear', NaN);
    row_px = interp1(z_ax_mm,  1:length(z_ax_mm),  ax_mm,  'linear', NaN);

    % Remove bubbles outside the image FOV
    valid = ~isnan(col_px) & ~isnan(row_px);
    gt_mm = gt_mm(valid, :);
    gt_px = [col_px(valid), row_px(valid)];
end
