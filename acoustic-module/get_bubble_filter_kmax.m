function kMax = get_bubble_filter_kmax(kgrid, Medium, isHybrid)
%GET_BUBBLE_FILTER_KMAX Return the cutoff used for bubble mass sources.

if ~isscalar(isHybrid)
    error('get_bubble_filter_kmax:InvalidHybridFlag', ...
        'isHybrid must be a scalar logical value.');
end
if logical(isHybrid)
    kMax = pi / kgrid.dt / Medium.SpeedOfSoundMinimum;
else
    kMax = kgrid.k_max;
end

end
