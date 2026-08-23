function run_log(action, varargin)
%RUN_LOG Compact run log: one line per frame, provenance banners once.
%
%   run_log('reset')                       start a run, clear all state
%   run_log('banner', key, fmt, args...)   print once per run for this key
%   run_log('stage', name, seconds)        accumulate time under a stage
%   run_log('frame', idx, last, seconds)   print the frame line, reset stages
%   run_log('summary', frames, seconds)    print the closing line
%
%   Stages accumulate, so a stage entered several times within one frame
%   (bubble-bubble interaction iterations) reports its total. The state is
%   persistent, which means a parfor worker keeps its own copy; the GPU path
%   runs serially, so the per-frame line stays complete there.

persistent stageNames stageSeconds bannerKeys

if isempty(stageNames)
    stageNames = {};
    stageSeconds = [];
    bannerKeys = {};
end

switch action
    case 'reset'
        stageNames = {};
        stageSeconds = [];
        bannerKeys = {};

    case 'banner'
        key = varargin{1};
        if any(strcmp(bannerKeys, key))
            return
        end
        bannerKeys{end+1} = key; %#ok<AGROW>
        fprintf(['=== ' varargin{2} ' ===\n'], varargin{3:end});

    case 'stage'
        name = varargin{1};
        seconds = varargin{2};
        slot = find(strcmp(stageNames, name), 1);
        if isempty(slot)
            stageNames{end+1} = name; %#ok<AGROW>
            stageSeconds(end+1) = seconds; %#ok<AGROW>
        else
            stageSeconds(slot) = stageSeconds(slot) + seconds;
        end

    case 'frame'
        % Fixed reading order; anything unlisted keeps its arrival order.
        order = {'MB', 'ODE', 'prop', 'RF'};
        rank = numel(order) + (1:numel(stageNames));
        for i = 1:numel(stageNames)
            slot = find(strcmp(order, stageNames{i}), 1);
            if ~isempty(slot)
                rank(i) = slot;
            end
        end
        [~, shown] = sort(rank);
        parts = cell(1, numel(shown));
        for i = 1:numel(shown)
            parts{i} = sprintf('%s %.2f', ...
                stageNames{shown(i)}, stageSeconds(shown(i)));
        end
        fprintf('frame %3d/%d | %s | total %.1f s\n', ...
            varargin{1}, varargin{2}, strjoin(parts, ' | '), varargin{3});
        stageNames = {};
        stageSeconds = [];

    case 'summary'
        frames = varargin{1};
        seconds = varargin{2};
        perFrame = seconds / max(1, frames);
        fprintf('=== %d frames | %.1f s | %.1f s/frame ===\n', ...
            frames, seconds, perFrame);

    otherwise
        error('run_log:UnknownAction', 'Unknown run_log action: %s.', action);
end

end
