function combined_out = preflight_transmit_record(n_mb_mask, n_trans_mask, ...
    n_mb_time, n_transducer_time, combined_requested, bubble_counts, run_param)
%PREFLIGHT_TRANSMIT_RECORD Size the transmit record and choose the path.
%
%   combined = preflight_transmit_record(...) reports what each transmit path
%   would cost in disk and in host memory, returns the path to actually take,
%   and errors if neither fits the disk.
%
%   preflight_array_limits sizes the receive path, all of it shaped by the
%   transducer. Nothing sized the transmit, and the transmit is what fails
%   first: the microbubble sensor is the union of every bubble position in
%   the batch, and with tiling that union is 54x the transducer mask. The
%   v11 run of 2026-09-03 asked the binary for a 550 GB output file on a
%   ~145 GB disk and found out 36 minutes in, at "Cannot write into dataset".
%
%   Sizing the disk alone was not enough, and left a hole worth naming. The
%   combined path reads its whole record into host memory -- run_simulation
%   ends in h5read inside kspaceFirstOrder3DC -- and then takes two subsets
%   off it before releasing it, so its peak is roughly 1.5x the record. The
%   split path never materialises its record at all: run_simulation_to_disk
%   leaves it in the binary's output file and project_transmit_to_bubbles
%   streams it. So a record between the host's memory and the disk's free
%   space passed this check and died hours later out of memory -- on the
%   combined path, which is the default.
%
%   Hence the choice is made here rather than from the setting alone. The
%   split path is never larger than the combined one on either axis: its
%   record covers the one-way window instead of the round trip, and it holds
%   none of it. So falling back is always available and always cheaper in
%   space; it costs one extra k-Wave run per pulse per batch (449.50 s
%   measured on run_20260827_082616). The setting still decides what to ask
%   for; the budgets decide what is affordable.
%
%   Budgets come from run_param.FreeDiskBytes and run_param.HostMemoryBytes
%   when given -- NaN says "cannot tell", and is reported, never enforced --
%   otherwise from the filesystem and from /proc and the cgroup on Linux.
%
%   Called on bubble_counts rather than on the built projection, so it runs
%   between the union mask and build_bubble_projection rather than after it.
%   The projection walks every frame of the batch, which is ~25 min at v11's
%   scale, and a batch that cannot be recorded should not pay for it.

bytesPerSample = 4;   % the binaries write float32

n_pulses   = size(bubble_counts, 2);
total_rows = sum(bubble_counts(:));

% Held on either path: the projection matrices, and one row per bubble per
% frame of projected pressure. Megabytes against the record's gigabytes, but
% they are what the frame loop actually reads, so they belong in the peak.
stencil    = (2 * run_param.MicrobubbleDeltaTruncation + 1)^3;
nonzeros_  = total_rows * stencil;
projection_bytes = nonzeros_ * 12 + n_mb_mask * 8 * n_pulses;
sensed_bytes     = total_rows * n_mb_time * bytesPerSample;
carried_bytes    = projection_bytes + sensed_bytes;

% The transducer's own record is held through the frame loop on both paths,
% one per pulse. It is the transducer mask over the round trip, which is the
% small one -- the microbubble mask is the 54x.
transducer_bytes = n_trans_mask * n_transducer_time * bytesPerSample * n_pulses;

% Combined: one sensor over the union of both masks, recorded for the round
% trip the transducer needs, read whole. extract_sensor_subset then takes the
% transducer rows and the microbubble rows off it before it is cleared, so
% the record and the larger subset are live at the same moment.
combined_record = (n_mb_mask + n_trans_mask) * n_transducer_time * bytesPerSample;
combined_peak   = combined_record ...
                + n_mb_mask * n_mb_time * bytesPerSample ...
                + transducer_bytes + carried_bytes;

% Split: the microbubble record covers the one-way window and is never read
% whole. What is held is a block of it at a time, which the projection sizes.
split_record = n_mb_mask * n_mb_time * bytesPerSample;
split_peak   = transducer_bytes + carried_bytes;

[free_disk, disk_given] = free_disk_bytes(run_param);
[free_host, host_given] = free_host_memory_bytes(run_param);

% Whether there is anything to fall back to. The split path streams its record
% out of the binary's own output file, so a solver that writes no such file
% ('3D', 'MATLAB' -- sim_setup gives them no BINARY_PATH) cannot take it at
% any size. validate_transmit_path_supported has already refused the runs
% where the split path would have been mandatory, so what is left here is a
% combined transmit that is the only path there is.
streaming = isfield(run_param, 'BINARY_PATH') && ~isempty(run_param.BINARY_PATH);

% The input file is written first and deleted once the binary has read it, so
% it has to fit alongside the output. It is grid-shaped rather than record-
% shaped, and a tenth of the output is a generous allowance at the sizes this
% check exists for.
combined_disk_need = combined_record * 1.1;
split_disk_need    = split_record * 1.1;

fprintf('=== Transmit-record preflight ===\n');
fprintf('  microbubble sensor points     %d (%.1fx the transducer''s %d)\n', ...
    n_mb_mask, n_mb_mask / max(n_trans_mask, 1), n_trans_mask);
fprintf('  combined path   record %7.1f GB   host peak %7.1f GB\n', ...
    combined_record / 2^30, combined_peak / 2^30);
fprintf('  split path      record %7.1f GB   host peak %7.1f GB\n', ...
    split_record / 2^30, split_peak / 2^30);
fprintf('  projection matrices           %.1f GB, %.3g nonzeros\n', ...
    projection_bytes / 2^30, nonzeros_);
fprintf('  projected pressure held       %.1f GB\n', sensed_bytes / 2^30);
fprintf('  free disk                     %s\n', ...
    describe_budget(free_disk, disk_given, data_path_name(run_param)));
fprintf('  free host memory              %s\n', ...
    describe_budget(free_host, host_given, '/proc and cgroup'));

% ------------------------------------------------------------ the choice
combined_out = combined_requested;
refusal = '';
if combined_requested
    if fits(combined_disk_need, free_disk) == false
        refusal = sprintf('record needs %.1f GB, %.1f GB of disk free', ...
            combined_disk_need / 2^30, free_disk / 2^30);
    elseif fits(combined_peak, free_host) == false
        refusal = sprintf('host peak %.1f GB, %.1f GB of memory free', ...
            combined_peak / 2^30, free_host / 2^30);
    end
    if ~isempty(refusal) && streaming
        combined_out = false;
    end
end

if combined_out
    path_name    = 'combined transducer + microbubble';
    chosen_disk  = combined_disk_need;
    chosen_peak  = combined_peak;
else
    path_name    = 'microbubble-only (split)';
    chosen_disk  = split_disk_need;
    chosen_peak  = split_peak;
end
fprintf('  path chosen                   %s\n', path_name);
if ~isempty(refusal) && ~combined_out
    fprintf('    combined refused: %s\n', refusal);
    fprintf('    falling back costs one k-Wave run per pulse per batch\n');
elseif ~isempty(refusal)
    fprintf('    WARNING: combined is over budget (%s)\n', refusal);
    fprintf(['             and there is no k-Wave output file to stream a ' ...
             'fallback from.\n']);
end

% ------------------------------------------------------------ the refusal
if fits(chosen_disk, free_disk) == false
    if combined_out
        % Only reachable with no binary to stream a fallback with; otherwise
        % the choice above has already taken the smaller path.
        no_room = ['There is no k-Wave output file to stream the split ' ...
                   'path from, so this is the only path.'];
    else
        no_room = ['The split path is the smaller of the two and does not ' ...
                   'fit either.'];
    end
    error('preflight_transmit_record:RecordTooLarge', ...
        ['The %s transmit record needs %.1f GB and %.1f GB is free at %s.\n' ...
         '%s\n' ...
         'What moves it: MicrobubbleDeltaTruncation (%d now, the stencil is ' ...
         '(2*th+1)^3), CombineTransmitSensors (%d now; false records the ' ...
         'bubbles over the one-way window instead of the round trip), ' ...
         'TransmitBatchSize, Microbubble.Number, Tiling.NumTiles.'], ...
        path_name, chosen_disk / 2^30, free_disk / 2^30, ...
        data_path_name(run_param), no_room, ...
        run_param.MicrobubbleDeltaTruncation, ...
        run_param.CombineTransmitSensors);
end

% Reported, not enforced. Once the split path is chosen there is nothing
% smaller to fall back to, and the estimate leaves out what MATLAB holds
% outside this accounting -- refusing on it would turn a run that fits into
% a run that is not attempted. The same convention preflight_array_limits
% uses for the receive path.
if fits(chosen_peak, free_host) == false
    fprintf(['  WARNING: the chosen path peaks at %.1f GB against a %.1f GB ' ...
             'budget.\n           There is no smaller path; the knobs above ' ...
             'are what move it.\n'], chosen_peak / 2^30, free_host / 2^30);
end

end


function ok = fits(need, budget)
% Three-valued on purpose: true, false, or [] when the budget is unknown.
% An unknown budget must not decide anything, so callers test == false.
if isnan(budget)
    ok = [];
else
    ok = need <= budget;
end
end


function s = describe_budget(bytes, given, where)
if isnan(bytes)
    s = 'unknown';
elseif given
    s = sprintf('%.1f GB (given, not measured)', bytes / 2^30);
else
    s = sprintf('%.1f GB at %s', bytes / 2^30, where);
end
end


function where = data_path_name(run_param)
where = '(no data path set)';
if isfield(run_param, 'DATA_PATH') && ~isempty(run_param.DATA_PATH)
    where = run_param.DATA_PATH;
end
end


function [bytes, from_override] = free_disk_bytes(run_param)
% Free space where the binary will write, or NaN if it cannot be asked.

bytes = NaN;
from_override = false;
% An explicit budget wins over the filesystem: a caller may know the disk is
% shared, and a test needs to say what "free" means without a real disk.
if isfield(run_param, 'FreeDiskBytes') && ~isempty(run_param.FreeDiskBytes)
    bytes = double(run_param.FreeDiskBytes);
    from_override = true;
    return
end
if ~isfield(run_param, 'DATA_PATH') || isempty(run_param.DATA_PATH)
    return
end
try
    f = java.io.File(run_param.DATA_PATH);
    if ~f.exists()
        f = java.io.File(fileparts(run_param.DATA_PATH));
    end
    usable = f.getUsableSpace();
    if usable > 0
        bytes = double(usable);
    end
catch
    % No JVM, or a path the JVM cannot see. Reported as unknown rather than
    % blocking a run that would otherwise be fine.
end

end


function [bytes, from_override] = free_host_memory_bytes(run_param)
% Host memory the process may still take, or NaN if it cannot be asked.
%
% Two ceilings, and the lower one binds. MemAvailable is the kernel's own
% estimate of what a new allocation can have without swapping. The cgroup's
% headroom is what a container is allowed regardless of that -- which is the
% one that binds on Colab, where /proc reports the whole machine.

bytes = NaN;
from_override = false;
if isfield(run_param, 'HostMemoryBytes') && ~isempty(run_param.HostMemoryBytes)
    bytes = double(run_param.HostMemoryBytes);
    from_override = true;
    return
end

ceilings = [];
try
    kb = regexp(fileread('/proc/meminfo'), ...
        'MemAvailable:\s+(\d+) kB', 'tokens', 'once');
    if ~isempty(kb)
        ceilings(end+1) = str2double(kb{1}) * 1024; %#ok<AGROW>
    end
catch
    % Not Linux, or /proc not mounted.
end
try
    % cgroup v2. "max" parses to NaN, which is the unlimited case and drops
    % out below with everything else that could not be read.
    limit   = str2double(strtrim(fileread('/sys/fs/cgroup/memory.max')));
    current = str2double(strtrim(fileread('/sys/fs/cgroup/memory.current')));
    if isfinite(limit) && isfinite(current)
        ceilings(end+1) = limit - current; %#ok<AGROW>
    end
catch
    % cgroup v1, or no cgroup filesystem.
end

if ~isempty(ceilings)
    bytes = min(ceilings);
end

end
