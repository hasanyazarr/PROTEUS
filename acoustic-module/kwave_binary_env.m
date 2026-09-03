function env = kwave_binary_env(run_param)
%KWAVE_BINARY_ENV Shell prefix the k-Wave binary needs to start.
%
%   run_simulation_to_disk runs the binary itself rather than letting
%   kspaceFirstOrder3DC do it, so it has to reproduce that function's
%   environment. The part that actually matters is LD_LIBRARY_PATH: the
%   shipped toolbox clears it, and the Colab driver rewrites that line with
%   the runtime's CUDA layout before every run.
%
%   Taken from that same line rather than duplicated here, so the notebook's
%   patch keeps being the single source of truth. Note that the toolbox is
%   unzipped from Drive over the clone, so it cannot be fixed in the repo --
%   reading it back is the only way to stay in step with it.
%
%   run_param.CudaLibPath overrides, for a caller that would rather say it
%   outright than rely on the patch having happened.

if ~isunix
    env = '';
    return
end

libPath = '';
if isfield(run_param, 'CudaLibPath') && ~isempty(run_param.CudaLibPath)
    libPath = run_param.CudaLibPath;
else
    toolboxFile = which('kspaceFirstOrder3DC');
    if ~isempty(toolboxFile)
        src = fileread(toolboxFile);
        token = regexp(src, 'export LD_LIBRARY_PATH=([^;'']*);', ...
            'tokens', 'once');
        if ~isempty(token)
            libPath = strtrim(token{1});
        end
    end
end

% Same OpenMP settings kspaceFirstOrder3DC sets, so the binary is placed and
% bound the way it is on the cached path.
env = sprintf('export LD_LIBRARY_PATH=%s; OMP_PLACES=cores; OMP_PROC_BIND=SPREAD; ', ...
    libPath);

end
