function maxPhaseStep = resolve_gpu_rk4_max_phase_step(settings)
%RESOLVE_GPU_RK4_MAX_PHASE_STEP Resolve and validate the RK4 phase limit.

maxPhaseStep = 0.25;
if isstruct(settings)
    if isfield(settings, 'GPURK4MaxPhaseStep')
        maxPhaseStep = settings.GPURK4MaxPhaseStep;
    elseif isfield(settings, 'rk4MaxPhaseStep')
        maxPhaseStep = settings.rk4MaxPhaseStep;
    end
elseif ~isempty(settings)
    maxPhaseStep = settings;
end

if ~isscalar(maxPhaseStep) || ~isnumeric(maxPhaseStep) || ...
        ~isreal(maxPhaseStep) || ~isfinite(maxPhaseStep) || ...
        maxPhaseStep <= 0
    error('calcBubbleResponse_GPU:InvalidMaxPhaseStep', ...
        'GPURK4MaxPhaseStep must be a finite positive scalar.');
end

end
