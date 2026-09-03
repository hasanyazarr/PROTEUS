% The streamed projection must be the same arithmetic as the cached path.
%
% The frame loop used to take the batch's recorded transmit, select this
% frame's rows out of it, and multiply by this frame's interpolation
% weights. build_bubble_projection stacks every frame's weights into one
% matrix addressed against the batch mask, and project_transmit_to_bubbles
% streams the record through it. Same nonzeros, same column order, so the
% result must be identical bit for bit -- not close, identical. If it ever
% is only close, the refactor stopped being a refactor.

clear Gq Aq Gm folderq nfq npq unionq Wq oldq newq newfq h5q recq

Gq = struct('Nx',36,'Ny',34,'Nz',32,'dx',1e-4,'dy',1e-4,'dz',1e-4, ...
            'sensor_on_grid',false);
Gq.x = (0:Gq.Nx-1)*Gq.dx;
Gq.y = (0:Gq.Ny-1)*Gq.dy;
Gq.z = (0:Gq.Nz-1)*Gq.dz;
Gq.full_size = [Gq.Nx; Gq.Ny; Gq.Nz];

Gm = struct('BoundingBox', struct('Center', zeros(3,1)), ...
            'Rotation', eye(3), 'Center', zeros(3,1));

nfq = 4;    % frames
npq = 2;    % pulses
folderq = tempname; mkdir(folderq);

rng(11)
for fq = 1:nfq
    Frame = struct();
    for pq = 1:npq
        nb = 5 + pq;                                  % different per pulse
        sub = [randi([8 28],nb,1) randi([8 26],nb,1) randi([8 24],nb,1)];
        pts = [Gq.x(sub(:,1))' Gq.y(sub(:,2))' Gq.z(sub(:,3))'] + 0.29*Gq.dx;
        Frame.(sprintf('Pulse%d',pq)) = struct( ...
            'Points', pts, 'Radius', (1:nb)'*1e-6, ...
            'Velocity', rand(nb,3)*1e-3, 'TileID', (1:nb)');
    end
    save(fullfile(folderq, sprintf('Frame_%d.mat', fq)), 'Frame');
end

Aq = struct('NumberOfFrames', nfq, 'StartFrame', 1, 'EndFrame', nfq);

for thq = [4 2]

    % --- the batch mask, exactly as main_RF builds it ---
    [sensor_batch, ~, ~, ~] = define_sensor_MB_all( ...
        Gq, folderq, Aq, npq, Gm, thq);
    unionq = find(logical(sensor_batch.mask));

    Wq = build_bubble_projection(Gq, folderq, Aq, npq, Gm, unionq, thq);

    % A record over the batch mask, single precision as the binaries write.
    nt = 17;
    recq = single(randn(numel(unionq), nt));

    for pq = 1:npq

        % --- the path this replaces ---
        oldq = [];
        for fq = 1:nfq
            MB = load_microbubbles(folderq, fq, pq, Gm, nfq);
            [sf, wf, ~, ~] = define_sensor_MB(Gq, MB, thq);
            rows = locate_in_sorted(unionq, find(sf.mask));
            sub  = recq(rows, 1:nt);
            oldq = [oldq; cast(full(wf*double(sub)), class(sub))]; %#ok<AGROW>
        end

        % --- in memory ---
        newq = project_transmit_to_bubbles(recq, Wq(pq).W, nt, 'single');
        assert(isequal(newq, oldq), ...
            'in-memory projection differs at th=%d pulse %d', thq, pq)

        % --- streamed from an HDF5 file, in more than one block ---
        h5q = [tempname '.h5'];
        h5create(h5q, '/p', size(recq), 'Datatype', 'single');
        h5write(h5q, '/p', recq);
        % Both regimes: several blocks, and one block that holds it all.
        % A budget change must not quietly stop exercising the split.
        newfq = project_transmit_to_bubbles(h5q, Wq(pq).W, nt, 'single', ...
            numel(unionq)*8*5);      % 5 samples per block, so 4 blocks
        assert(isequal(newfq, oldq), ...
            'streamed projection differs at th=%d pulse %d', thq, pq)
        newfq = project_transmit_to_bubbles(h5q, Wq(pq).W, nt, 'single', ...
            numel(unionq)*8*nt*4);   % one block
        assert(isequal(newfq, oldq), ...
            'single-block streamed projection differs at th=%d pulse %d', thq, pq)
        delete(h5q);

        % --- the row offsets address the right frame ---
        for fq = 1:nfq
            r = Wq(pq).RowFirst(fq):Wq(pq).RowLast(fq);
            MB = load_microbubbles(folderq, fq, pq, Gm, nfq);
            [~, wf, MBk, ~] = define_sensor_MB(Gq, MB, thq);
            assert(numel(r) == size(wf,1), 'row block must be one row per bubble')
            assert(isequal(Wq(pq).MB{fq}.points, MBk.points), ...
                'cached bubbles must be the frame''s own')
            assert(size(newq(r,:),1) == size(MBk.points,1))
        end
    end

    % The stencil actually changed between the two passes of this loop.
    fprintf('  th=%d: union %d points, %d nonzeros in pulse 1\n', ...
        thq, numel(unionq), nnz(Wq(1).W));
end

rmdir(folderq, 's');
disp('test_transmit_projection_equivalence: all assertions passed')
