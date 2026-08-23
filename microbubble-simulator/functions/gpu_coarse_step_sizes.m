function hInterval = gpu_coarse_step_sizes(idxCoarse, dt, nSub)
%GPU_COARSE_STEP_SIZES RK4 substep size for every coarse output interval.
%   Appending the final sample to the coarse grid can leave a last interval
%   narrower than the stride. Every interval therefore gets the step its own
%   width implies, so the integration lands exactly on each output time
%   instead of running past the end of the pulse.
%
%   IDXCOARSE are the sample indices of the coarse grid, DT the
%   nondimensional step between two neighbouring samples, and NSUB the number
%   of substeps taken per interval.

if ~isvector(idxCoarse) || numel(idxCoarse) < 2
    error('calcBubbleResponse_GPU:InvalidCoarseGrid', ...
        'The coarse grid needs at least two samples.');
end
spans = diff(double(idxCoarse(:)'));
if any(spans <= 0)
    error('calcBubbleResponse_GPU:InvalidCoarseGrid', ...
        'The coarse grid must be strictly increasing.');
end
if ~isscalar(nSub) || nSub < 1 || nSub ~= floor(nSub)
    error('calcBubbleResponse_GPU:InvalidSubstepCount', ...
        'The substep count must be a positive integer.');
end

hInterval = cast(spans, 'like', dt) * dt / cast(nSub, 'like', dt);
end
