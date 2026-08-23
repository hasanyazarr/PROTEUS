function [nSub, info] = calculate_gpu_rk4_substeps(...
    eqparam, pulse, maxPhaseStep, stride)
%CALCULATE_GPU_RK4_SUBSTEPS Select RK4 substeps from the fastest timescale.

maxPhaseStep = resolve_gpu_rk4_max_phase_step(maxPhaseStep);
if nargin < 4
    stride = 1;
end
if ~isscalar(stride) || ~isnumeric(stride) || ~isreal(stride) || ...
        ~isfinite(stride) || stride < 1 || stride ~= floor(stride)
    error('calcBubbleResponse_GPU:InvalidStride', ...
        'The output stride must be a positive integer.');
end

naturalFrequencies = [eqparam.omega_0];
if isempty(naturalFrequencies) || ~isreal(naturalFrequencies) || ...
        any(~isfinite(naturalFrequencies)) || any(naturalFrequencies <= 0)
    error('calcBubbleResponse_GPU:InvalidNaturalFrequency', ...
        'All bubble natural angular frequencies must be finite and positive.');
end
if ~isscalar(pulse.w) || ~isnumeric(pulse.w) || ~isreal(pulse.w) || ...
        ~isfinite(pulse.w) || pulse.w < 0
    error('calcBubbleResponse_GPU:InvalidTransmitFrequency', ...
        'pulse.w must be a finite non-negative angular frequency.');
end

dimensionalStep = double(stride) * (pulse.tq(2) - pulse.tq(1));
dampingRates = zeros(size(naturalFrequencies));
if all(isfield(eqparam, 'delta'))
    dampingRates = abs([eqparam.delta]) .* naturalFrequencies;
end
maxAngularFrequency = max(...
    [naturalFrequencies, dampingRates, pulse.w], [], 'all');
nSub = max(1, ceil(maxAngularFrequency * dimensionalStep / maxPhaseStep));

info.substeps = nSub;
info.maxAngularFrequency = maxAngularFrequency;
info.maxPhaseStep = maxPhaseStep;
info.dimensionalStep = dimensionalStep;
info.actualPhaseStep = maxAngularFrequency * dimensionalStep / nSub;

end
