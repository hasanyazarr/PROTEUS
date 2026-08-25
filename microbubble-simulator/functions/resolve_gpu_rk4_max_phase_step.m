function maxPhaseStep = resolve_gpu_rk4_max_phase_step(settings)
%RESOLVE_GPU_RK4_MAX_PHASE_STEP Resolve and validate the RK4 phase limit.
%
% The default is the largest step the sweep of 2026-08-25 could not tell
% apart from a finer one. Against the error-controlled CPU reference the
% worst mass-source disagreement was 6.2e-4 at 0.25 rad and 4.5e-4 at 0.5,
% both inside the ~1e-4 floor that single precision and the pressure
% interpolant impose regardless of step size. 1.0 rad leaves that floor at
% 3.0e-3, and does it at the small-bubble, high-frequency corner the
% production radii occupy, so the sweep stops here.

maxPhaseStep = 0.5;
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
