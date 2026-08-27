repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'scripts'));

% A throwaway settings file and results folder for the whole file.
tmp = tempname;
mkdir(tmp);
cleanupTmp = onCleanup(@() rmdir(tmp, 's'));

settingsPath = fullfile(tmp, 'settings.mat');
Acquisition = struct('NumberOfFrames', 50, 'StartFrame', 1, 'EndFrame', 50);
SimulationParameters = struct('Solver', '3DG', 'SamplingRate', 110240000);
Microbubble = struct('Number', 200, 'UseGPU', true);
save(settingsPath, 'Acquisition', 'SimulationParameters', 'Microbubble');

%% The manifest names the code, the settings, and the environment.
savedir = fullfile(tmp, 'run_a');
manifestPath = write_run_manifest(savedir, settingsPath, ...
    struct('StartFrame', 1, 'EndFrame', 25));

assert(isfile(manifestPath), 'no manifest written');
M = jsondecode(fileread(manifestPath));

assert(isfield(M, 'Simulator'), 'no Simulator block');
assert(~isempty(M.Simulator.Commit), 'empty commit');
assert(islogical(M.Simulator.Dirty) || ischar(M.Simulator.Dirty), ...
    'Dirty must be a boolean or the string ''unavailable''');
assert(~isempty(M.Settings.SHA256), 'no settings checksum');
assert(M.Settings.Values.Microbubble.Number == 200, 'settings not embedded');
assert(strcmp(M.Environment.Solver, '3DG'), 'solver not recorded');

%% The settings file lands beside the frames, so the folder stands alone.
copied = fullfile(savedir, 'settings_used.mat');
assert(isfile(copied), 'settings_used.mat missing');
S = load(copied, 'Microbubble');
assert(S.Microbubble.Number == 200, 'copied settings do not match');

%% A second call appends a segment instead of replacing the first.
% Batched and resumed runs call main_RF more than once; overwriting would
% hide that the run was resumed at all.
write_run_manifest(savedir, settingsPath, ...
    struct('StartFrame', 26, 'EndFrame', 50));
M = jsondecode(fileread(manifestPath));

assert(numel(M.Segments) == 2, ...
    sprintf('expected 2 segments, got %d', numel(M.Segments)));
starts = arrayfun(@(s) s.StartFrame, M.Segments);
assert(isequal(sort(starts(:))', [1 26]), 'segment frames not preserved');

%% A driver block written after the fact survives a resume.
M.Driver = struct('NOTEBOOK_NAME', 'test.ipynb');
fid = fopen(manifestPath, 'w');
fprintf(fid, '%s', jsonencode(M));
fclose(fid);

write_run_manifest(savedir, settingsPath, ...
    struct('StartFrame', 51, 'EndFrame', 60));
M = jsondecode(fileread(manifestPath));

assert(isfield(M, 'Driver'), 'the resume erased the driver block');
assert(strcmp(M.Driver.NOTEBOOK_NAME, 'test.ipynb'), 'driver block corrupted');
assert(numel(M.Segments) == 3, 'segment lost on resume');

%% An unreadable manifest is replaced, not propagated.
savedirB = fullfile(tmp, 'run_b');
mkdir(savedirB);
fid = fopen(fullfile(savedirB, 'run_manifest.json'), 'w');
fprintf(fid, 'not json at all');
fclose(fid);

manifestPathB = write_run_manifest(savedirB, settingsPath, struct());
M = jsondecode(fileread(manifestPathB));
assert(numel(M.Segments) == 1, 'a corrupt manifest should start over');

disp('test_write_run_manifest: all assertions passed');
