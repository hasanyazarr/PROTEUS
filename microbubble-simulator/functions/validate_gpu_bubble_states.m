function validate_gpu_bubble_states(...
    displacement, velocity, intermediateNonPositive)
%VALIDATE_GPU_BUBBLE_STATES Reject invalid GPU states without clamping.

finalComplex = any(imag(displacement) ~= 0, 1) | ...
    any(imag(velocity) ~= 0, 1);
invalidBubbles = find(finalComplex);
if ~isempty(invalidBubbles)
    error('calcBubbleResponse_GPU:ComplexState', ...
        'GPU RK4 integration produced a complex state for local bubble(s): %s.', ...
        format_indices(invalidBubbles));
end
if any(~isfinite(displacement), 'all') || any(~isfinite(velocity), 'all')
    invalidBubbles = find(any(~isfinite(displacement), 1) | ...
        any(~isfinite(velocity), 1));
    error('calcBubbleResponse_GPU:NonFiniteState', ...
        'GPU RK4 integration produced a non-finite state for local bubble(s): %s.', ...
        format_indices(invalidBubbles));
end

finalNonPositive = any(1 + displacement <= 0, 1);
invalidBubbles = find(intermediateNonPositive | finalNonPositive);
if ~isempty(invalidBubbles)
    error('calcBubbleResponse_GPU:NonPositiveRadius', ...
        'GPU RK4 integration produced a non-positive radius for local bubble(s): %s.', ...
        format_indices(invalidBubbles));
end

end

function output = format_indices(indices)
if isempty(indices)
    output = 'unknown';
else
    output = strjoin(string(indices), ',');
end
end
