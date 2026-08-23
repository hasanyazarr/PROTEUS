function shellModel = validate_gpu_bubble_solver_inputs(pulse, shell)
%VALIDATE_GPU_BUBBLE_SOLVER_INPUTS Validate the fixed-grid GPU contract.

requiredPulseFields = {'t', 'tq', 'p', 'w'};
for fieldIndex = 1:numel(requiredPulseFields)
    if ~isfield(pulse, requiredPulseFields{fieldIndex})
        error('calcBubbleResponse_GPU:MissingPulseField', ...
            'pulse.%s is required.', requiredPulseFields{fieldIndex});
    end
end
if ~isvector(pulse.t) || ~isvector(pulse.tq) || numel(pulse.t) < 2 || ...
        ~isreal(pulse.t) || ~isreal(pulse.tq) || ...
        any(~isfinite(pulse.t)) || any(~isfinite(pulse.tq))
    error('calcBubbleResponse_GPU:InvalidTimeGrid', ...
        'pulse.t and pulse.tq must be finite real vectors with at least two samples.');
end
if ~isequal(pulse.t(:), pulse.tq(:))
    error('calcBubbleResponse_GPU:MismatchedTimeGrids', ...
        'GPU integration requires pulse.t and pulse.tq to be identical.');
end
if size(pulse.p, 2) ~= numel(pulse.t)
    error('calcBubbleResponse_GPU:PressureLengthMismatch', ...
        'The number of pressure samples must equal the time-grid length.');
end
if size(pulse.p, 1) ~= numel(shell)
    error('calcBubbleResponse_GPU:PressureBubbleCountMismatch', ...
        'The number of pressure rows must equal the number of bubbles.');
end

timeSteps = diff(pulse.t(:));
stepTolerance = max(1, max(abs(timeSteps))) * eps(class(timeSteps)) * 16;
if any(timeSteps <= 0) || any(abs(timeSteps - timeSteps(1)) > stepTolerance)
    error('calcBubbleResponse_GPU:NonUniformTimeGrid', ...
        'GPU integration requires a strictly increasing uniform time grid.');
end

if isempty(shell) || ~all(isfield(shell, 'model'))
    error('calcBubbleResponse_GPU:MissingShellModel', ...
        'Every bubble must define a shell model.');
end
shellModel = shell(1).model;
if ~all(strcmp({shell.model}, shellModel))
    error('calcBubbleResponse_GPU:MixedShellModels', ...
        'All microbubbles in a GPU batch must use the same shell model.');
end

switch shellModel
    case 'Marmottant'
        return
    case 'Segers'
        referenceCoefficients = shell(1).coeff;
        for shellIndex = 2:numel(shell)
            if ~isequaln(shell(shellIndex).coeff, referenceCoefficients)
                error('calcBubbleResponse_GPU:InconsistentShellCoefficients', ...
                    'All Segers shells in a GPU batch must use the same coefficients.');
            end
        end
    case 'SegersTable'
        referenceGrid = shell(1).sig.GridVectors{1};
        referenceValues = shell(1).sig.Values;
        for shellIndex = 2:numel(shell)
            candidateGrid = shell(shellIndex).sig.GridVectors{1};
            candidateValues = shell(shellIndex).sig.Values;
            if ~isequaln(candidateGrid, referenceGrid) || ...
                    ~isequaln(candidateValues, referenceValues)
                error('calcBubbleResponse_GPU:InconsistentShellTables', ...
                    'All SegersTable shells in a GPU batch must use the same lookup table.');
            end
        end
    otherwise
        error('calcBubbleResponse_GPU:UnsupportedShellModel', ...
            'Unsupported GPU shell model: %s.', shellModel);
end

end
