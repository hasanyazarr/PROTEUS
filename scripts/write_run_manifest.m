function manifestPath = write_run_manifest(savedir, settingsfile, Segment)
%WRITE_RUN_MANIFEST  Record what produced a run, next to the run's own data.
%
% A run's identity lives in three places and no single one of them is enough
% to reproduce it: the simulator commit (Colab clones the fork, so the commit
% says which code ran), the effective settings (the driver overrides fields in
% the .mat before running), and the driver's own knobs (which never reach any
% settings file). This function writes the first two into
% <savedir>/run_manifest.json and copies the settings file to
% <savedir>/settings_used.mat, so the results folder is self-sufficient before
% anything copies it anywhere. The driver appends its own block afterwards.
%
% Args:
%   savedir       Folder the RF frames are written to
%   settingsfile  Full path to the settings .mat actually being used
%   Segment       Optional struct with StartFrame and EndFrame. One is
%                 appended per call, so a batched or resumed run keeps the
%                 record of every segment rather than only the last.
%
% Returns:
%   manifestPath  Full path to the manifest written

if nargin < 3 || isempty(Segment)
    Segment = struct();
end

SCHEMA_VERSION = 1;

if ~exist(savedir, 'dir')
    mkdir(savedir)
end
manifestPath = fullfile(savedir, 'run_manifest.json');

repoRoot = fileparts(fileparts(mfilename('fullpath')));

%==========================================================================
% SIMULATOR: WHICH CODE RAN
%
% Settings alone cannot answer this. Two v7 runs with matching geometry
% fields placed their bubbles differently because the streamline module had
% changed in between, and nothing in either run recorded that.
%==========================================================================
Manifest.Simulator.Root = repoRoot;
Manifest.Simulator.Commit = git_query(repoRoot, 'rev-parse HEAD');
Manifest.Simulator.Branch = git_query(repoRoot, 'rev-parse --abbrev-ref HEAD');
Manifest.Simulator.Describe = git_query(repoRoot, 'describe --always --dirty');

porcelain = git_query(repoRoot, 'status --porcelain');
if strcmp(porcelain, 'unavailable')
    % An absent field reads the same as a clean tree. Say which it is.
    Manifest.Simulator.Dirty = 'unavailable';
    Manifest.Simulator.DirtyFiles = {};
else
    dirtyFiles = strtrim(strsplit(porcelain, newline));
    dirtyFiles = dirtyFiles(~cellfun(@isempty, dirtyFiles));
    Manifest.Simulator.Dirty = ~isempty(dirtyFiles);
    Manifest.Simulator.DirtyFiles = dirtyFiles;
end

%==========================================================================
% SETTINGS: WHICH PARAMETERS RAN
%
% The values are embedded, not just referenced, because the .mat is the part
% that goes missing. A checksum ties the embedded copy to the file.
%==========================================================================
Manifest.Settings.File = settingsfile;
Manifest.Settings.SHA256 = file_sha256(settingsfile);

S = load(settingsfile);
try
    Manifest.Settings.Values = S;
    jsonencode(Manifest.Settings.Values);   % fails here, not at write time
catch err
    Manifest.Settings.Values = struct();
    Manifest.Settings.ValuesError = err.message;
end

copyfile(settingsfile, fullfile(savedir, 'settings_used.mat'));

%==========================================================================
% ENVIRONMENT
%==========================================================================
Manifest.Environment.MatlabVersion = version;
Manifest.Environment.Computer = computer;
Manifest.Environment.Host = hostname();
Manifest.Environment.GPU = gpu_name();
if isfield(S, 'SimulationParameters') && ...
        isfield(S.SimulationParameters, 'Solver')
    Manifest.Environment.Solver = S.SimulationParameters.Solver;
else
    Manifest.Environment.Solver = 'unavailable';
end

%==========================================================================
% SEGMENTS
%
% main_RF runs once per transmit batch and once more per resume, so the
% manifest accumulates rather than overwrites.
%==========================================================================
Segment.WrittenAt = utc_timestamp();
if ~isfield(Segment, 'StartFrame'), Segment.StartFrame = []; end
if ~isfield(Segment, 'EndFrame'),   Segment.EndFrame = [];   end

priorSegments = {};
if isfile(manifestPath)
    try
        prior = jsondecode(fileread(manifestPath));
        if isfield(prior, 'Segments')
            priorSegments = as_cell(prior.Segments);
        end
        % The driver appends its own block after this file is first written;
        % a resume must not erase it.
        if isfield(prior, 'Driver')
            Manifest.Driver = prior.Driver;
        end
    catch
        % An unreadable manifest is replaced, not merged.
    end
end
Manifest.Segments = [priorSegments, {Segment}];

Manifest.SchemaVersion = SCHEMA_VERSION;
Manifest.WrittenAt = Segment.WrittenAt;

fid = fopen(manifestPath, 'w');
if fid == -1
    error('write_run_manifest:CannotWrite', ...
        'Could not open %s for writing.', manifestPath);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(Manifest, 'PrettyPrint', true));

fprintf('Run manifest written to: %s\n', manifestPath);
end


function out = git_query(repoRoot, args)
%GIT_QUERY  Run one git command, or report that git could not answer.
cmd = sprintf('git -C "%s" %s', repoRoot, args);
[status, result] = system(cmd);
if status ~= 0
    out = 'unavailable';
else
    out = strtrim(result);
end
end


function out = file_sha256(filepath)
try
    digest = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(filepath, 'r');
    cleanup = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
    digest.update(bytes);
    % Width 2 is forced: dec2hex sizes to the widest value in the array,
    % so a hash whose bytes all fall below 0x10 would lose its padding.
    out = lower(reshape(dec2hex(typecast(digest.digest(), 'uint8'), 2)', 1, []));
catch
    out = 'unavailable';
end
end


function out = hostname()
try
    out = char(java.net.InetAddress.getLocalHost().getHostName());
catch
    out = 'unavailable';
end
end


function out = gpu_name()
try
    d = gpuDevice();
    out = d.Name;
catch
    out = 'unavailable';
end
end


function out = utc_timestamp()
out = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
end


function out = as_cell(value)
%AS_CELL  jsondecode returns a struct for a one-element array and a struct
%array for a homogeneous one; both have to end up as a cell array.
if iscell(value)
    out = value(:)';
elseif isstruct(value)
    out = num2cell(value(:)');
else
    out = {};
end
end
