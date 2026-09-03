function stats = validate_tile_placement(TileCfg, vtuStruct, Geometry)
% VALIDATE_TILE_PLACEMENT  Refuse tile offsets that push the vessel out of the domain.
%
% Every tile transform is known before the first frame, so where the tiled
% vessel lands is checkable for free. run_20260708_043218 shipped 0.30% of its
% labelled microbubbles outside the simulated domain and 0.8% inside the
% transmit ringdown because nothing looked: those bubbles carry ground-truth
% labels with no RF behind them.
%
% The bound depends on whether the tile rotation is randomised. With rotation
% on, theta is drawn uniformly per tile, so the check has to hold for every
% angle: the vessel sweeps a disc about Geometry.Center whose radius is the
% largest in-plane distance from that centre to any vessel cell. With rotation
% off the vessel keeps its own footprint and the axis-aligned extent is used.
% Elevation never sweeps -- the rotation is about that axis -- so its extent is
% the vessel's own either way.
%
% INPUT:
%  - TileCfg   : struct with .Enabled, .RandomizeRotation and .Transforms
%                (as built by build_tile_transforms; .Offset is in the vessel
%                frame, so T_image = Geometry.Rotation * Offset).
%  - vtuStruct : struct with .points (Ncells x 3, m, canonical vessel frame).
%  - Geometry  : struct with .Rotation, .BoundingBox.Center, .Center and
%                .Domain (.Xmin/.Xmax depth, .Ymin/.Ymax lateral,
%                .Zmin/.Zmax elevation, m, image frame).
%
% OUTPUT:
%  - stats : struct with .NumTiles, .MaxOverhang (m; <= 0 when every tile fits),
%            .WorstTile and .WorstAxis.
%
% Errors with generate_streamlines:TilePlacementOutsideDomain when any tile
% overhangs, naming the tile, the axis and the overhang in mm.

stats = struct('NumTiles', 0, 'MaxOverhang', -Inf, 'WorstTile', 0, ...
    'WorstAxis', '');

if ~TileCfg.Enabled
    return
end

R  = Geometry.Rotation;
bb = Geometry.BoundingBox.Center(:);
c  = Geometry.Center(:);
D  = Geometry.Domain;

img = (R * (vtuStruct.points.' - bb) + c).';   % Ncells x 3, [depth width elev]
rel = img - c.';                               % about the rotation centre

if TileCfg.RandomizeRotation
    % Worst case over all theta: the swept disc.
    radius = max(hypot(rel(:,1), rel(:,2)));
    inPlaneLo = [-radius, -radius];
    inPlaneHi = [ radius,  radius];
else
    inPlaneLo = min(rel(:,1:2), [], 1);
    inPlaneHi = max(rel(:,1:2), [], 1);
end
lo = [inPlaneLo, min(rel(:,3))] + c.';
hi = [inPlaneHi, max(rel(:,3))] + c.';

boxLo = [D.Xmin, D.Ymin, D.Zmin];
boxHi = [D.Xmax, D.Ymax, D.Zmax];
axisNames = {'depth', 'lateral', 'elevation'};

transforms = TileCfg.Transforms;
stats.NumTiles = numel(transforms);

for k = 1:numel(transforms)
    T = (R * transforms(k).Offset(:)).';        % vessel-frame offset -> image
    over = max(boxLo - (lo + T), (hi + T) - boxHi);
    [worst, axisIdx] = max(over);
    if worst > stats.MaxOverhang
        stats.MaxOverhang = worst;
        stats.WorstTile   = k;
        stats.WorstAxis   = axisNames{axisIdx};
    end
end

if stats.MaxOverhang > 0
    error('generate_streamlines:TilePlacementOutsideDomain', ...
        ['Tile %d leaves the simulated domain by %.2f mm in %s. Bubbles ' ...
         'there are labelled in the ground truth with no RF behind them. ' ...
         'Narrow Acquisition.Tiling.%s, or move Geometry.startDepth.'], ...
        stats.WorstTile, 1e3*stats.MaxOverhang, stats.WorstAxis, ...
        range_field_for(stats.WorstAxis));
end

end


function name = range_field_for(axisName)
switch axisName
    case 'depth',     name = 'DepthRange';
    case 'lateral',   name = 'WidthRange';
    otherwise,        name = 'ElevRange';
end
end
