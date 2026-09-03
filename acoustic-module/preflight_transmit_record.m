function preflight_transmit_record(n_mb_mask, n_trans_mask, ...
    n_mb_time, n_transducer_time, combined, Projection, run_param)
%PREFLIGHT_TRANSMIT_RECORD Refuse a transmit whose record will not fit.
%
%   preflight_array_limits sizes the receive path, all of it shaped by the
%   transducer. Nothing sized the transmit, and the transmit is what fails
%   first: the microbubble sensor is the union of every bubble position in
%   the batch, and with tiling that union is 54x the transducer mask. The
%   v11 run of 2026-09-03 asked the binary for a 550 GB output file on a
%   ~145 GB disk and found out 36 minutes in, at 25% of the first pulse,
%   as "Cannot write into dataset". Everything needed to say so in seconds
%   is known here.
%
%   Reported, not enforced, is the host memory the projection needs. MATLAB
%   on Linux cannot ask how much is free, so the number is for the reader --
%   the same convention preflight_array_limits uses.

bytesPerSample = 4;   % the binaries write float32

if combined
    record_rows = n_mb_mask + n_trans_mask;   % worst case, before overlap
    record_time = n_transducer_time;
    path_name   = 'combined transducer + microbubble';
else
    record_rows = n_mb_mask;
    record_time = n_mb_time;
    path_name   = 'microbubble-only';
end
record_bytes = record_rows * record_time * bytesPerSample;

W = Projection(1).W;
projection_bytes = 0;
for k = 1:numel(Projection)
    projection_bytes = projection_bytes + ...
        nnz(Projection(k).W) * 12 + ...          % value plus row index
        size(Projection(k).W, 2) * 8;            % column pointers
end
sensed_bytes = 0;
for k = 1:numel(Projection)
    sensed_bytes = sensed_bytes + size(Projection(k).W, 1) * n_mb_time * 4;
end

fprintf('=== Transmit-record preflight ===\n');
fprintf('  path                          %s\n', path_name);
fprintf('  microbubble sensor points     %d (%.1fx the transducer''s %d)\n', ...
    n_mb_mask, n_mb_mask / max(n_trans_mask, 1), n_trans_mask);
fprintf('  record the binary will write  %.1f GB  (%d x %d)\n', ...
    record_bytes / 2^30, record_rows, record_time);
fprintf('  projection matrices           %.1f GB, %d nonzeros\n', ...
    projection_bytes / 2^30, nnz(W) * numel(Projection));
fprintf('  projected pressure held       %.1f GB\n', sensed_bytes / 2^30);

where = '(no data path set)';
if isfield(run_param, 'DATA_PATH') && ~isempty(run_param.DATA_PATH)
    where = run_param.DATA_PATH;
end

[free_bytes, from_override] = free_disk_bytes(run_param);
if isnan(free_bytes)
    fprintf('  free disk                     unknown\n');
    return
end
if from_override
    fprintf('  free disk                     %.1f GB (given, not measured)\n', ...
        free_bytes / 2^30);
else
    fprintf('  free disk                     %.1f GB at %s\n', ...
        free_bytes / 2^30, where);
end

% The input file is written first and deleted once the binary has read it,
% so it has to fit alongside the output. It is grid-shaped rather than
% record-shaped, and a tenth of the output is a generous allowance at the
% sizes this check exists for.
needed = record_bytes * 1.1;
if needed > free_bytes
    error('preflight_transmit_record:RecordTooLarge', ...
        ['The %s transmit record needs %.1f GB and %.1f GB is free at %s.\n' ...
         'What moves it: MicrobubbleDeltaTruncation (%d now, the stencil is ' ...
         '(2*th+1)^3), CombineTransmitSensors (%d now; false records the ' ...
         'bubbles over the one-way window instead of the round trip), ' ...
         'TransmitBatchSize, Microbubble.Number, Tiling.NumTiles.'], ...
        path_name, needed / 2^30, free_bytes / 2^30, where, ...
        run_param.MicrobubbleDeltaTruncation, ...
        run_param.CombineTransmitSensors);
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
