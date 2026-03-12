% run_RF_on_Mac.m
% Colab proteus_data_generation.py akışına uygun: ground truth üretir,
% Solver = 3DC yapar, main_RF çalıştırır. Önce Command Window'da path_setup çalıştırın.
%
% Kullanım:
%   >> path_setup
%   >> run_RF_on_Mac

% Path'leri ekle (path_setup zaten çalıştırılmış olmalı)
PATHS = path_setup();
addpath(PATHS.Start);
addpath(PATHS.AcousticModulePath);
addpath(PATHS.StreamlineFunctions);
addpath(PATHS.GUIfunctions);

% Ayarlar dosyası (Colab'da my_simulation_settings.mat; Mac'te GUI_output_parameters_v2.mat)
SETTINGS_NAME = 'GUI_output_parameters_v4.mat';
% v2 workspace kökünde (data_generation_proteus); PROTEUS bir üst klasör
SETTINGS_PATH = fullfile(fileparts(PATHS.Start), SETTINGS_NAME);

% Her run için benzersiz timestamp'li klasör oluştur (karışmasın diye)
RUN_ID = datestr(now, 'yyyymmdd_HHMMss');
RUN_TAG = ['run_' RUN_ID];
GROUND_TRUTH_FOLDER = RUN_TAG;  % ground_truth_frames/run_YYYYMMDD_HHMMSS/
SAVE_FOLDER = RUN_TAG;          % RESULTS/run_YYYYMMDD_HHMMSS/
fprintf('=== RUN ID: %s ===\n', RUN_TAG);
fprintf('    Ground truth -> %s\n', fullfile(PATHS.GroundTruthPath, GROUND_TRUTH_FOLDER));
fprintf('    Results      -> %s\n', fullfile(PATHS.ResultsPath, SAVE_FOLDER));

% Bubble sayısı (Colab ile aynı olması için 100; 10 = varsayılan)
NUM_BUBBLES = 100;

% Frame sayısı (her frame = bir RF kaydı = bir B-mode görüntü; video için binlerce yapabilirsin)
% Varsayılan 10; video için örn. 100–1000+ (her frame ek süre getirir)
NUM_FRAMES = 10;

% 2D slice: true = Z'de tek dilim (grid Nx×Ny×1), ~100–200× hızlı; false = tam 3D
USE_2D_SLICE = false;

% ===== DOMAIN REDUCTION =====
% Domain'i küçülterek simülasyon süresini düşür.
% true = aşağıdaki ROI sınırları kullanılır; false = GUI'deki otomatik domain
REDUCE_DOMAIN = false;

% ROI sınırları (metre cinsinden). GUI'deki mm değerlerini 1e-3 ile çarp.
% Mevcut tam domain: X=[-1.2, 44.6] Y=[-15.4, 15.4] Z=[-9.2, 9.2] mm
% Damar merkezi ~(25, 0, 0) mm civarında.
% DİKKAT:
%   - Transducer x≈0'da! X_MIN ≤ -1.2 mm olmalı.
%   - 9L-D aperture = 44.1 mm → Y en az [-22.5, 22.5] mm olmalı!
%   - 9L-D element height = 6 mm → Z en az [-3.5, 3.5] mm olmalı!
%   - Gerçekçi küçültme: sadece X derinliği (X_MAX) azaltılabilir.
% Öneriler:
%   Test (~20 dk):   X=[-1.2, 25] Y=[-23,23] Z=[-5,5] mm
%   Orta (~40 dk):   X=[-1.2, 30] Y=[-23,23] Z=[-6,6] mm
%   Geniş (~1 saat): X=[-1.2, 35] Y=[-23,23] Z=[-8,8] mm
ROI_X_MIN = -0.0012;  % metre (= -1.2 mm) — transducer dahil olmalı!
ROI_X_MAX =  0.025;   % metre (= 25 mm)  — derinlik kısıtlaması
ROI_Y_MIN = -0.023;   % metre (= -23 mm) — tüm transducer aperture!
ROI_Y_MAX =  0.023;   % metre (= 23 mm)
ROI_Z_MIN = -0.005;   % metre (= -5 mm)  — element height (6mm) + margin
ROI_Z_MAX =  0.005;   % metre (= 5 mm)

% 1) subplus workaround (Curve Fitting Toolbox yoksa)
if ~exist('subplus', 'file')
    disp('Creating subplus workaround...');
    fid = fopen(fullfile(PATHS.Start, 'subplus.m'), 'w');
    fprintf(fid, 'function y = subplus(x)\n');
    fprintf(fid, '%% Workaround for Curve Fitting Toolbox subplus\n');
    fprintf(fid, 'y = max(x, 0);\n');
    fprintf(fid, 'end\n');
    fclose(fid);
    addpath(PATHS.Start);
    disp('subplus created.');
end

% 2) Bubble, frame sayısı ve (opsiyonel) 2D slice ayarla ve kaydet
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', 'Microbubble', ...
    'SimulationParameters', 'Transducer', 'Transmit');
Microbubble.Number = NUM_BUBBLES;
Acquisition.NumberOfFrames = NUM_FRAMES;
Acquisition.StartFrame = 1;
Acquisition.EndFrame = NUM_FRAMES;
% Always reload the original (unmodified) settings to get correct domain
ORIG_SETTINGS = fullfile(fileparts(PATHS.Start), 'GUI_output_parameters_v2.mat');
if exist(ORIG_SETTINGS, 'file') && ~strcmp(ORIG_SETTINGS, SETTINGS_PATH)
    orig = load(ORIG_SETTINGS, 'Geometry');
    Geometry.Domain.Zmin = orig.Geometry.Domain.Zmin;
    Geometry.Domain.Zmax = orig.Geometry.Domain.Zmax;
    Geometry.Domain.Manual = orig.Geometry.Domain.Manual;
    fprintf('=== Restored original Z domain: [%.6e, %.6e] ===\n', ...
        Geometry.Domain.Zmin, Geometry.Domain.Zmax);
end

% --- Domain Reduction ---
if REDUCE_DOMAIN
    Geometry.Domain.Manual = true;
    Geometry.Domain.Xmin = ROI_X_MIN;
    Geometry.Domain.Xmax = ROI_X_MAX;
    Geometry.Domain.Ymin = ROI_Y_MIN;
    Geometry.Domain.Ymax = ROI_Y_MAX;
    Geometry.Domain.Zmin = ROI_Z_MIN;
    Geometry.Domain.Zmax = ROI_Z_MAX;
    fprintf('=== REDUCED DOMAIN (Manual) ===\n');
    fprintf('    X: [%.1f, %.1f] mm\n', ROI_X_MIN*1e3, ROI_X_MAX*1e3);
    fprintf('    Y: [%.1f, %.1f] mm\n', ROI_Y_MIN*1e3, ROI_Y_MAX*1e3);
    fprintf('    Z: [%.1f, %.1f] mm\n', ROI_Z_MIN*1e3, ROI_Z_MAX*1e3);
    vol_mm3 = (ROI_X_MAX-ROI_X_MIN)*(ROI_Y_MAX-ROI_Y_MIN)*(ROI_Z_MAX-ROI_Z_MIN)*1e9;
    fprintf('    Volume: %.0f mm3\n', vol_mm3);
elseif USE_2D_SLICE %#ok<UNRCH>
    % Safety check for valid Z domain values
    if ~isfield(Geometry.Domain, 'Zmin') || ~isfield(Geometry.Domain, 'Zmax') %#ok<UNRCH>
        error('Geometry.Domain.Zmin or Zmax not defined');
    end
    if isnan(Geometry.Domain.Zmin) || isnan(Geometry.Domain.Zmax)
        error('Geometry.Domain.Zmin or Zmax is NaN');
    end
    if isinf(Geometry.Domain.Zmin) || isinf(Geometry.Domain.Zmax)
        error('Geometry.Domain.Zmin or Zmax is Inf');
    end

    zCenter = (Geometry.Domain.Zmin + Geometry.Domain.Zmax) / 2;
    fprintf('=== 2D slice: Z domain [%.6e, %.6e] -> center = %.6e m ===\n', ...
        Geometry.Domain.Zmin, Geometry.Domain.Zmax, zCenter);

    Geometry.Domain.Manual = true;
    Geometry.Domain.Zmin = zCenter;
    Geometry.Domain.Zmax = zCenter;
    fprintf('=== Grid will be Nx×Ny×1 (2D slice) ===\n');
else
    fprintf('=== Full 3D: Z domain [%.6e, %.6e] ===\n', ...
        Geometry.Domain.Zmin, Geometry.Domain.Zmax);
end
save(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', 'Microbubble', ...
    'SimulationParameters', 'Transducer', 'Transmit');
fprintf('=== Microbubble.Number = %d, NumberOfFrames = %d ===\n', NUM_BUBBLES, NUM_FRAMES);

% 3) Ground truth üret (Colab'daki "Create GT MB Positions" adımı)
disp('=== Generating ground truth (streamlines) ===');
loaded = load(SETTINGS_PATH);
generate_streamlines(loaded.Geometry, loaded.Microbubble, loaded.Acquisition, ...
    PATHS, GROUND_TRUTH_FOLDER, false);
disp('=== Ground truth generated ===');

% 4) Solver: 3DC uses rebuilt kspaceFirstOrder-OMP binary (fixed, no -ffast-math)
% The original binary produced NaN due to -ffast-math; rebuilt binary is NaN-free.
USE_MATLAB_SOLVER = false;  % 3DC binary is fixed; set true only for pure-MATLAB fallback
load(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', 'Microbubble', ...
    'SimulationParameters', 'Transducer', 'Transmit');
if USE_MATLAB_SOLVER
    disp('=== Setting Solver = MATLAB (avoids 3DC NaN on Mac) ==='); %#ok<UNRCH>
    SimulationParameters.Solver = 'MATLAB';
else
    disp('=== Setting Solver = 3DC ===');
    SimulationParameters.Solver = '3DC';
end
save(SETTINGS_PATH, 'Acquisition', 'Geometry', 'Medium', 'Microbubble', ...
    'SimulationParameters', 'Transducer', 'Transmit');

% simulation-settings klasöründeki kopyayı da güncelle (main_RF oradan okur)
copyfile(SETTINGS_PATH, fullfile(PATHS.SettingsPath, SETTINGS_NAME));

% 5) RF simülasyonunu çalıştır
disp('=== Running main_RF ===');
main_RF(SETTINGS_NAME, GROUND_TRUTH_FOLDER, SAVE_FOLDER);
disp('=== RF Simulation Complete ===');

% 6) Run bilgilerini kaydet (her run kendi klasöründe settings kopyası tutsun)
runResultsDir = fullfile(PATHS.ResultsPath, SAVE_FOLDER);
if ~exist(runResultsDir, 'dir'), mkdir(runResultsDir); end
copyfile(SETTINGS_PATH, fullfile(runResultsDir, SETTINGS_NAME));
run_info.RUN_ID = RUN_TAG;
run_info.timestamp = datestr(now);
run_info.NUM_BUBBLES = NUM_BUBBLES;
run_info.NUM_FRAMES = NUM_FRAMES;
run_info.USE_2D_SLICE = USE_2D_SLICE;
run_info.USE_MATLAB_SOLVER = USE_MATLAB_SOLVER;
run_info.SETTINGS_NAME = SETTINGS_NAME;
run_info.GROUND_TRUTH_FOLDER = GROUND_TRUTH_FOLDER;
run_info.SAVE_FOLDER = SAVE_FOLDER;
save(fullfile(runResultsDir, 'run_info.mat'), 'run_info');
fprintf('=== Run info saved to %s ===\n', runResultsDir);
