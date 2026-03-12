% visualize_bmode.m
% Visualize B-mode images from RF data frames using DAS.
%
% Usage:
%   >> path_setup
%   >> visualize_bmode

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

% Use the settings copy saved inside the run folder (exact config that produced the data)
SETTINGS_PATH = fullfile(RESULTS_FOLDER, SETTINGS_NAME);
if ~exist(SETTINGS_PATH, 'file')
    % Fallback: use the copy in simulation-settings
    SETTINGS_PATH = fullfile(PATHS.SettingsPath, SETTINGS_NAME);
    fprintf('Note: Using settings from %s\n', SETTINGS_PATH);
end

% Output subfolder inside the run
VIS_FOLDER = fullfile(RESULTS_FOLDER, 'bmode_vis');
if ~exist(VIS_FOLDER, 'dir'), mkdir(VIS_FOLDER); end
fprintf('B-mode visualizations will be saved to: %s\n', VIS_FOLDER);

dynamicRange = 60; % dB

if ~exist(RESULTS_FOLDER, 'dir')
    error('Results folder not found: %s', RESULTS_FOLDER);
end

% Quick check: is RF data in Frame_*.mat valid? (explains all-black if not)
fprintf('=== Checking RF data in first frame ===\n');
flist = dir(fullfile(RESULTS_FOLDER, 'Frame*.mat'));
if isempty(flist)
    error('No Frame*.mat files in %s', RESULTS_FOLDER);
end
load(fullfile(flist(1).folder, flist(1).name), 'RF');
if iscell(RF)
    rf1 = RF{1};
else
    rf1 = RF;
end
fprintf('  RF size: %s, range: [%.2e, %.2e], NaN: %d, all zero: %d\n', ...
    mat2str(size(rf1)), min(rf1(:)), max(rf1(:)), ...
    sum(isnan(rf1(:))), all(rf1(:) == 0));
if all(rf1(:) == 0) || all(isnan(rf1(:)))
    warning('RF data in %s is all zeros or NaN. B-mode will be black.', flist(1).name);
    fprintf('  Possible causes: (1) Simulation (main_RF) produced no signal; (2) Pulsing scheme mismatch.\n');
    fprintf('  Run diagnose_data.m for details, or re-run run_RF_on_Mac.m to regenerate RF.\n');
end

fprintf('=== Performing Delay-and-Sum Reconstruction ===\n');
fprintf('Loading frames from: %s\n', RESULTS_FOLDER);
[IMG, z, x] = DAS_reconstruction(RESULTS_FOLDER, SETTINGS_PATH);

% IMG from DAS is (Nx, Nz, Nframes); for imagesc(x,z,C) we need C = (Nz, Nx)
Nframes = size(IMG, 3);
fprintf('Reconstructed %d frames\n', Nframes);
fprintf('  IMG (log) range: [%.2f, %.2f] dB, NaN count: %d\n', ...
    min(IMG(:)), max(IMG(:)), sum(isnan(IMG(:))));
if all(isnan(IMG(:))) || (max(IMG(:)) < -dynamicRange)
    warning('DAS output is all NaN or below display range. B-mode will be black.');
end

% Transpose so rows = axial (z), cols = lateral (x); clip for display
IMG_display = permute(IMG, [2 1 3]);
IMG_display(isnan(IMG_display)) = -dynamicRange;
IMG_display(IMG_display < -dynamicRange) = -dynamicRange;

% 1) Save all B-mode PNGs
fprintf('Saving B-mode PNGs...\n');
npad = length(num2str(Nframes));
for iframe = 1:Nframes
    fname = fullfile(VIS_FOLDER, sprintf('Bmode_Frame_%s.png', num2str(iframe, ['%0' num2str(npad) 'd'])));
    fig_vis = figure('Visible', 'off');
    imagesc(x*1e3, z*1e3, IMG_display(:,:,iframe), [-dynamicRange 0]);
    colormap(gray);
    axis image;
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    title(sprintf('Frame %d', iframe));
    saveas(fig_vis, fname);
    close(fig_vis);
    if mod(iframe, 20) == 0 || iframe == Nframes
        fprintf('  Saved %d / %d\n', iframe, Nframes);
    end
end
fprintf('B-mode images saved to: %s\n', VIS_FOLDER);

% 2) One figure with slider to browse frames
fig_browse = figure('Name', 'B-mode frames');
ax = axes(fig_browse);
h_img = imagesc(ax, x*1e3, z*1e3, IMG_display(:,:,1), [-dynamicRange 0]);
colormap(ax, gray);
axis(ax, 'image');
xlabel(ax, 'Lateral (mm)');
ylabel(ax, 'Axial (mm)');
title(ax, sprintf('Frame 1 / %d', Nframes));
colorbar(ax);

if Nframes > 1
    h_slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', Nframes, 'Value', 1, ...
        'Position', [20 10 400 20], 'SliderStep', [1/max(1,Nframes-1) 10/max(1,Nframes-1)], ...
        'Callback', @(~,~) updateFrame());
    uicontrol('Style', 'text', 'Position', [430 10 80 20], 'String', 'Frame');
end

function updateFrame()
if Nframes == 1, return; end
k = round(get(h_slider, 'Value'));
k = max(1, min(Nframes, k));
set(h_img, 'CData', IMG_display(:,:,k));
title(ax, sprintf('Frame %d / %d', k, Nframes));
end

% 3) Sample grid (up to 20 frames)
Ngrid = min(20, Nframes);
idx_grid = round(linspace(1, Nframes, Ngrid));
figure('Name', 'B-mode sample grid');
[ny, nx] = subplot_grid(Ngrid);
for i = 1:Ngrid
    subplot(ny, nx, i);
    imagesc(x*1e3, z*1e3, IMG_display(:,:,idx_grid(i)), [-dynamicRange 0]);
    colormap(gca, gray);
    axis image;
    title(sprintf('F%d', idx_grid(i)));
end
sgtitle(sprintf('Sample of %d frames', Ngrid));
saveas(gcf, fullfile(VIS_FOLDER, 'Bmode_Sample_Grid.png'));

fprintf('=== Visualization complete ===\n');
fprintf('All visualizations saved to: %s\n', VIS_FOLDER);

function [ny, nx] = subplot_grid(N)
nx = ceil(sqrt(N));
ny = ceil(N / nx);
end
