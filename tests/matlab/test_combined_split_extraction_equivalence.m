% The combined transmit's two subsets must be the two records the split
% transmit would have produced.
%
% Since 2026-09-05 the preflight moves a run between these paths on its own,
% on budgets rather than on a setting, so "they agree" stopped being a claim
% about a choice someone made deliberately and became a claim the code relies
% on. This is that claim, for the half of it that is ours.
%
% What is ours: which rows come out and in what order. k-Wave records sensor
% points in linear-index order, so a run over one mask returns find(mask)'s
% rows in ascending order. extract_sensor_subset has to give back exactly
% those rows, in exactly that order, out of a record made over the union.
%
% What is NOT ours, and is not asserted here: that recording over the round
% trip and truncating to the one-way window equals recording over the one-way
% window. That is causality in the time stepping -- a later sample cannot
% change an earlier one -- and asserting it needs the solver, which does not
% run on this machine.

clear Gx maskT maskM idxC idxT idxM Px sdx subT subM nT nM rngx trial

nT = 20;    % round-trip window, what the transducer needs
nM = 9;     % one-way window, all the bubbles ever read

%% The transducer and microbubble rows come back exactly as recorded alone.
maskT = false(6,5,4);  maskT([3 11 40 57 92]) = true;
maskM = false(6,5,4);  maskM([7 11 23 40 66 99 108]) = true;   % 11, 40 shared

idxC = find(maskT | maskM);
idxT = find(maskT);
idxM = find(maskM);

% A field with a distinct value per (point, sample), so a misplaced row or a
% transposed order cannot pass by coincidence. single, as the binaries write.
Px  = single((1:numel(maskT))' * 1000 + (0:nT-1));
sdx.p = Px(idxC, :);                       % what the combined run records

subT = extract_sensor_subset(sdx, idxC, idxT, nT);
subM = extract_sensor_subset(sdx, idxC, idxM, nM);

assert(isequal(subT.p, Px(idxT, 1:nT)), ...
    'the transducer subset is not what a transducer-only run would record');
assert(isequal(subM.p, Px(idxM, 1:nM)), ...
    'the microbubble subset is not what an MB-only run would record');
assert(isa(subT.p, 'single') && isa(subM.p, 'single'), ...
    'extraction must not promote the record');

%% Shared points are read by both, not consumed by the first.
% The union stores a point carried by both masks once; both extractions have
% to find it. Rows 11 and 40 above are the case.
assert(any(ismember(idxT, idxM)), 'the fixture lost its overlap');
assert(isequal(subT.p(idxT == 11, :), Px(11, 1:nT)));
assert(isequal(subM.p(idxM == 11, :), Px(11, 1:nM)));

%% Disjoint masks, and one mask contained in the other.
maskM = false(6,5,4);  maskM([2 8 44]) = true;          % disjoint from maskT
idxC = find(maskT | maskM);
sdx.p = Px(idxC, :);
assert(isequal(extract_sensor_subset(sdx, idxC, find(maskM), nM).p, ...
    Px(find(maskM), 1:nM))); %#ok<FNDSB>

maskM = maskT;                                           % identical masks
idxC = find(maskT | maskM);
sdx.p = Px(idxC, :);
assert(isequal(extract_sensor_subset(sdx, idxC, idxT, nT).p, Px(idxT, 1:nT)));

%% The same holds for masks drawn at random, which is where an order
%% assumption would break if it were wrong.
rng(4)
for trial = 1:200
    maskT = rand(6,5,4) < 0.25;
    maskM = rand(6,5,4) < 0.4;
    if ~any(maskT(:)) || ~any(maskM(:)); continue; end
    idxC = find(maskT | maskM);
    sdx.p = Px(idxC, :);
    assert(isequal(extract_sensor_subset(sdx, idxC, find(maskT), nT).p, ...
        Px(find(maskT), 1:nT)), 'trial %d: transducer rows', trial); %#ok<FNDSB>
    assert(isequal(extract_sensor_subset(sdx, idxC, find(maskM), nM).p, ...
        Px(find(maskM), 1:nM)), 'trial %d: microbubble rows', trial); %#ok<FNDSB>
end

%% A target the record does not carry is an error, not a shorter answer.
% intersect would quietly return the rows it found. The frame loop would then
% multiply a projection built for N bubbles by a record of fewer, and the
% shapes happen to be forgiving enough that it could go unnoticed.
maskT = false(6,5,4);  maskT([3 11 40]) = true;
idxC = find(maskT);
sdx.p = Px(idxC, :);
okx = false;
try
    extract_sensor_subset(sdx, idxC, [3; 11; 40; 41], nT);
    okx = true;
catch ex
    assert(strcmp(ex.identifier, 'extract_sensor_subset:TargetNotRecorded'), ...
        'wrong error: %s', ex.identifier)
end
assert(~okx, 'a target outside the record must not come back truncated')

disp('test_combined_split_extraction_equivalence: all assertions passed')
