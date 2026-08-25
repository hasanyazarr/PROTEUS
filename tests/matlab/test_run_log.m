repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'acoustic-module'));

%% The frame line nests ODE inside MB and orders the stages for reading.
run_log('reset');
run_log('stage', 'RF', 1.4);
run_log('stage', 'ODE', 47.6);
run_log('stage', 'prop', 5.9);
run_log('stage', 'MB', 47.8);
run_log('count', 'MB', 200);
line = evalc('run_log(''frame'', 2, 500, 59.4)');
assert(contains(line, 'frame   2/500'), line);
assert(contains(line, '200 MB'), line);
assert(contains(line, 'MB 47.8 (ODE 47.6)'), line);
% ODE measures part of MB, so it must not appear as a sibling column that
% invites summing the row.
assert(~contains(line, '| ODE'), line);
% Declared reading order, whatever order the stages arrived in.
assert(strfind(line, 'MB 47.8') < strfind(line, 'prop 5.9'), line);
assert(strfind(line, 'prop 5.9') < strfind(line, 'RF 1.4'), line);

%% Stages and counts reset after a frame; banner state does not.
run_log('banner', 'solver', 'MB solver: GPU-RK4');
first = evalc('run_log(''banner'', ''solver'', ''MB solver: GPU-RK4'')');
assert(isempty(first), first);
line = evalc('run_log(''frame'', 3, 500, 10)');
assert(~contains(line, 'MB'), line);
assert(~contains(line, 'prop'), line);

%% Banners survive a function clear, which a path change triggers.
run_log('reset');
shown = evalc('run_log(''banner'', ''solver'', ''once'')');
assert(contains(shown, 'once'), shown);
clear functions
repeated = evalc('run_log(''banner'', ''solver'', ''once'')');
assert(isempty(repeated), repeated);

%% A repeated count overwrites instead of accumulating.
run_log('reset');
run_log('count', 'MB', 200);
run_log('count', 'MB', 200);
line = evalc('run_log(''frame'', 1, 10, 1)');
assert(numel(strfind(line, '200 MB')) == 1, line);

%% The ETA averages over the frames done, and is dropped on the last frame.
run_log('reset');
evalc('run_log(''frame'', 1, 3, 60)');
line = evalc('run_log(''frame'', 2, 3, 120)');
% Mean of 60 and 120 is 90 s, one frame left.
assert(contains(line, 'ETA 1m30s'), line);
last = evalc('run_log(''frame'', 3, 3, 60)');
assert(~contains(last, 'ETA'), last);

%% Durations read as hours, minutes, or seconds.
run_log('reset');
evalc('run_log(''frame'', 1, 501, 60)');
line = evalc('run_log(''frame'', 2, 501, 60)');
assert(contains(line, 'ETA 8h19m'), line);

%% An unknown stage still prints, after the ones with a declared order.
run_log('reset');
run_log('stage', 'save', 0.5);
run_log('stage', 'MB', 1.0);
line = evalc('run_log(''frame'', 1, 2, 1.5)');
assert(strfind(line, 'MB 1.0') < strfind(line, 'save 0.5'), line);

%% A frame with no stages at all is still one legible line.
run_log('reset');
line = evalc('run_log(''frame'', 1, 1, 2)');
assert(contains(line, 'frame 1/1'), line);
assert(contains(line, '2.0s'), line);

%% The summary reports the wall clock and the per-frame cost.
run_log('reset');
line = evalc('run_log(''summary'', 500, 30000)');
assert(contains(line, '500 frames'), line);
assert(contains(line, '8h20m'), line);
assert(contains(line, '60.0 s/frame'), line);

%% An unknown action is rejected rather than silently ignored.
try
    run_log('nonsense');
    error('test_run_log:NoError', 'Expected run_log to reject the action.');
catch exception
    assert(strcmp(exception.identifier, 'run_log:UnknownAction'), ...
        exception.identifier);
end

disp('test_run_log: all assertions passed');
