function [vtuStruct, stats] = crop_vessel_to_slab(vtuStruct, Geometry, halfThickness)
% CROP_VESSEL_TO_SLAB  Restrict microbubble seeding to the imaged elevation slab.
%
% Zeroes the per-cell seeding weight (vtuStruct.density) for vessel cells whose
% image-frame elevation falls further than halfThickness from the elevation
% centre, so a seeded microbubble starts inside the slice the transducer
% actually images instead of somewhere in the elevation blur.
%
% This is the single largest lever on in-plane microbubble density. The
% renal_tree geometry spans 6.5 mm in elevation against a slab of roughly
% 1 mm, so about 96% of any bubble budget is spent out of plane: measured on
% 2026-09-03, clipping to the slab multiplies the in-plane count by 21, while
% raising Microbubble.Number from 200 to 1200 multiplies it by 6. See
% docs/vessel_tiling.md.
%
% Unlike CROP_VESSEL_TO_DOMAIN this may run *before* vessel tiling. A tile
% rotates about the elevation axis and offsets it by at most
% Acquisition.Tiling.ElevRange, so a cell's elevation survives the transform
% and the crop stays correct afterwards. Depth and lateral mix under that same
% rotation, which is why the domain crop cannot be composed with tiling.
%
% Operates on the density weights only (not row deletion), so vtuStruct.points,
% vtuStruct.velocities and the Grid lookup all stay valid.
%
% INPUT:
%  - vtuStruct    : struct with .points (Ncells x 3, m, canonical vessel frame)
%                   and .density (Ncells x 1, per-cell seeding weight).
%  - Geometry     : struct with .Rotation (3x3), .BoundingBox.Center (3x1, m)
%                   and .Center (3x1, m). The slab is centred on the image-frame
%                   elevation of Geometry.Center.
%  - halfThickness: half the slab thickness, m.
%
% OUTPUT:
%  - vtuStruct : same struct with .density zeroed outside the slab, renormalized.
%  - stats     : struct with .NTotal, .NInSlab, .InSlabFraction, .SeedableCells
%                and .HalfThickness.

if ~isfield(vtuStruct, 'density')
    error('crop_vessel_to_slab:noDensity', ...
        'vtuStruct.density missing; run apply_velocity_weighted_seeding first.');
end

% Vessel -> image transform: image = R*(vessel - bb) + c. Only the elevation
% row is needed, but the full product keeps this readable next to the domain
% crop it mirrors.
R  = Geometry.Rotation;
bb = Geometry.BoundingBox.Center(:);
c  = Geometry.Center(:);
img = (R * (vtuStruct.points.' - bb) + c).';   % Ncells x 3, [depth width elev]

inSlab = abs(img(:,3) - c(3)) <= halfThickness;

density = vtuStruct.density;
density(~inSlab) = 0;

if sum(density) == 0
    error('crop_vessel_to_slab:empty', ...
        ['No seedable vessel cells inside the %.3g mm elevation slab. ' ...
         'Widen Acquisition.ElevationSlab or check Geometry.Center.'], ...
        2e3*halfThickness);
end

vtuStruct.density = density / sum(density);

stats = struct();
stats.NTotal        = numel(inSlab);
stats.NInSlab       = nnz(inSlab);
stats.InSlabFraction = mean(inSlab);
stats.SeedableCells = nnz(vtuStruct.density > 0);
stats.HalfThickness = halfThickness;

end
