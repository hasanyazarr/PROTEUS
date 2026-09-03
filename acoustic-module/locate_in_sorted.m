function loc = locate_in_sorted(haystack, needles)
%LOCATE_IN_SORTED Positions of needles within the ascending list haystack.
%
% This was intersect(haystack, needles), which sorts both every call. That
% is free when haystack is one frame's own mask and ruinous when it is the
% union over a whole batch: 150000 bubbles against a 2.5e6-entry union is
% 150000 sorts of the union. Both lists come out of find() or sub2ind over
% an ndgrid, so both are already ascending, and a binary search over the
% edges costs O(|needles| log |haystack|) with no sort at all.
%
% intersect silently drops a needle that is not in the haystack, which
% would misalign the weights against the indices; every caller here builds
% the mask from these very points, so a miss is a bug and says so.
%
% Returns a column, as intersect did, because the caller transposes it.

loc = discretize(needles(:), [haystack(:); Inf]);
if any(isnan(loc)) || any(haystack(loc) ~= needles(:))
    error('locate_in_sorted:NotFound', ...
        ['A sensor point is not in the mask its weights are addressed ' ...
         'against. The weight columns and the recorded rows would not ' ...
         'line up.'])
end

end
