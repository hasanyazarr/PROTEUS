function run_log(action, varargin)
%RUN_LOG Compact run log: one line per frame, provenance banners once.
%
%   run_log('reset')                       start a run, clear all state
%   run_log('banner', key, fmt, args...)   print once per run for this key
%   run_log('count', name, value)          per-frame count shown on the line
%   run_log('stage', name, seconds)        accumulate time under a stage
%   run_log('frame', idx, last, seconds)   print the frame line, reset stages
%   run_log('summary', frames, seconds)    print the closing line
%
%   Stages accumulate, so a stage entered several times within one frame
%   (bubble-bubble interaction iterations) reports its total. Stages listed
%   in NESTED are printed inside their parent's parentheses, because they
%   measure part of the parent rather than time alongside it -- a flat list
%   invites summing columns that overlap.
%
%   The state lives in root appdata rather than in a persistent variable.
%   Persistent variables are cleared whenever MATLAB drops a function from
%   memory, which the acquisition loop used to trigger every frame by
%   changing the search path; that reset the banner keys and reprinted every
%   banner per frame. Root appdata survives it. A parfor worker still keeps
%   its own copy, but the GPU path runs serially, so the per-frame line stays
%   complete there.

STATE_KEY = 'PROTEUS_run_log';

% Reading order for the top-level stages; anything unlisted keeps its
% arrival order after these.
ORDER = {'MB', 'prop', 'RF'};

% Stages that measure part of another stage, as {child, parent}.
NESTED = {'ODE', 'MB'};

% A state left behind by an earlier version of this file would be missing
% fields, so check the shape rather than only whether something is there.
state = getappdata(0, STATE_KEY);
if ~isstruct(state) || ~isfield(state, 'framesDone')
    state = empty_state();
end

switch action
    case 'reset'
        state = empty_state();

    case 'banner'
        key = varargin{1};
        if any(strcmp(state.bannerKeys, key))
            return
        end
        state.bannerKeys{end+1} = key;
        fprintf(['=== ' varargin{2} ' ===\n'], varargin{3:end});

    case 'count'
        % Overwrites rather than accumulates: the bubble-bubble interaction
        % loop reports the same count once per iteration.
        name = varargin{1};
        slot = find(strcmp(state.countNames, name), 1);
        if isempty(slot)
            state.countNames{end+1} = name;
            state.countValues(end+1) = varargin{2};
        else
            state.countValues(slot) = varargin{2};
        end

    case 'stage'
        name = varargin{1};
        seconds = varargin{2};
        slot = find(strcmp(state.stageNames, name), 1);
        if isempty(slot)
            state.stageNames{end+1} = name;
            state.stageSeconds(end+1) = seconds;
        else
            state.stageSeconds(slot) = state.stageSeconds(slot) + seconds;
        end

    case 'frame'
        frameIdx = varargin{1};
        lastFrame = varargin{2};
        seconds = varargin{3};

        state.framesDone = state.framesDone + 1;
        state.secondsDone = state.secondsDone + seconds;

        % Right-align the frame number so the columns line up for the run.
        frameFormat = sprintf('frame %%%dd/%%d', numel(num2str(lastFrame)));
        parts = {sprintf(frameFormat, frameIdx, lastFrame)};
        parts = [parts, format_counts(state)];
        parts = [parts, format_stages(state, ORDER, NESTED)];
        parts{end+1} = sprintf('%.1fs', seconds);

        remaining = lastFrame - frameIdx;
        if remaining > 0
            meanSeconds = state.secondsDone / state.framesDone;
            parts{end+1} = sprintf('ETA %s', ...
                format_duration(remaining * meanSeconds));
        end

        fprintf('%s\n', strjoin(parts, ' | '));

        state.stageNames = {};
        state.stageSeconds = [];
        state.countNames = {};
        state.countValues = [];

    case 'summary'
        frames = varargin{1};
        seconds = varargin{2};
        perFrame = seconds / max(1, frames);
        fprintf('=== %d frames | %s | %.1f s/frame ===\n', ...
            frames, format_duration(seconds), perFrame);

    otherwise
        error('run_log:UnknownAction', 'Unknown run_log action: %s.', action);
end

setappdata(0, STATE_KEY, state);

end


function state = empty_state()
state.stageNames = {};
state.stageSeconds = [];
state.countNames = {};
state.countValues = [];
state.bannerKeys = {};
state.framesDone = 0;
state.secondsDone = 0;
end


function parts = format_counts(state)
parts = cell(1, numel(state.countNames));
for i = 1:numel(state.countNames)
    parts{i} = sprintf('%d %s', ...
        state.countValues(i), state.countNames{i});
end
end


function parts = format_stages(state, order, nested)
% Rank the top-level stages by the reading order, leaving the nested ones
% out of the ranking: they are printed inside their parent instead.
isNested = ismember(state.stageNames, nested(:, 1)');
topLevel = find(~isNested);

rank = numel(order) + (1:numel(topLevel));
for i = 1:numel(topLevel)
    slot = find(strcmp(order, state.stageNames{topLevel(i)}), 1);
    if ~isempty(slot)
        rank(i) = slot;
    end
end
[~, byRank] = sort(rank);
shown = topLevel(byRank);

parts = cell(1, numel(shown));
for i = 1:numel(shown)
    name = state.stageNames{shown(i)};
    parts{i} = sprintf('%s %.1f', name, state.stageSeconds(shown(i)));

    inner = children_of(state, name, nested);
    if ~isempty(inner)
        parts{i} = sprintf('%s (%s)', parts{i}, strjoin(inner, ', '));
    end
end
end


function inner = children_of(state, parent, nested)
% Format the recorded stages nested under PARENT, in the order NESTED
% declares them.
inner = {};
for i = 1:size(nested, 1)
    if ~strcmp(nested{i, 2}, parent)
        continue
    end
    slot = find(strcmp(state.stageNames, nested{i, 1}), 1);
    if ~isempty(slot)
        inner{end+1} = sprintf('%s %.1f', ...
            state.stageNames{slot}, state.stageSeconds(slot)); %#ok<AGROW>
    end
end
end


function text = format_duration(seconds)
seconds = max(0, round(seconds));
hours = floor(seconds / 3600);
minutes = floor(mod(seconds, 3600) / 60);
if hours > 0
    text = sprintf('%dh%02dm', hours, minutes);
elseif minutes > 0
    text = sprintf('%dm%02ds', minutes, mod(seconds, 60));
else
    text = sprintf('%ds', seconds);
end
end
