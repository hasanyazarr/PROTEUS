function [seed, source] = resolve_random_seed(requested, label)
%RESOLVE_RANDOM_SEED Start a random stream from a settings field and say which.
%
% Accepts a nonnegative integer, or 'shuffle' to let MATLAB draw one from the
% clock. Either way the return value is the integer the stream was actually
% started from, so a run that pinned nothing can still be replayed from what
% it recorded -- which is the difference between attribution and
% reproducibility.
%
%   [seed, source] = RESOLVE_RANDOM_SEED(requested)
%   [seed, source] = RESOLVE_RANDOM_SEED(requested, label)
%
% label names the settings field in error messages.

if nargin < 2 || isempty(label)
    label = 'RandomSeed';
end

if isempty(requested)
    requested = 'shuffle';
end

if ischar(requested) || isstring(requested)
    if ~strcmpi(char(requested), 'shuffle')
        error('resolve_random_seed:InvalidSeed', ...
            '%s must be a nonnegative integer or ''shuffle'', got ''%s''.', ...
            label, char(requested));
    end
    rng('shuffle');
    state = rng;
    seed = double(state.Seed);
    source = 'shuffle';
else
    if ~isnumeric(requested) || ~isscalar(requested) || ...
            ~isfinite(requested) || requested < 0 || ...
            requested ~= floor(requested) || requested >= 2^32
        error('resolve_random_seed:InvalidSeed', ...
            ['%s must be a nonnegative integer below 2^32 or ' ...
             '''shuffle''.'], label);
    end
    seed = double(requested);
    source = 'settings';
end

rng(seed, 'twister');

end
