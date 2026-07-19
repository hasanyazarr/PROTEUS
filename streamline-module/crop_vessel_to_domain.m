function [vtuStruct, stats] = crop_vessel_to_domain(vtuStruct, Geometry, marginFrac)
% CROP_VESSEL_TO_DOMAIN  Restrict microbubble seeding to the simulated domain.
%
% Zeroes the per-cell seeding weight (vtuStruct.density) for vessel cells whose
% image-frame position falls outside the simulated domain box (Geometry.Domain),
% so every seeded microbubble starts inside the region that is actually
% simulated and imaged. This generalizes MB placement to any probe / frequency /
% FOV with no hand-tuning of tiling offset ranges: the box comes straight from
% the settings (Geometry.Domain), which already encodes the elevation slab
% thickness, lateral aperture, and axial clip for the current configuration.
%
% Operates on the density weights only (not row deletion), so vtuStruct.points,
% vtuStruct.velocities and the Grid lookup all stay valid. Requires
% vtuStruct.density, which apply_velocity_weighted_seeding produces.
%
% INPUT:
%  - vtuStruct : struct with .points (Ncells x 3, m, canonical vessel frame),
%                .density (Ncells x 1, per-cell seeding weight).
%  - Geometry  : struct with .Rotation (3x3), .BoundingBox.Center (3x1, m),
%                .Center (3x1, m), and .Domain (.Xmin/.Xmax depth,
%                .Ymin/.Ymax lateral, .Zmin/.Zmax elevation, m, image frame).
%  - marginFrac: fraction of each axis span to shrink the box by (default 0.15),
%                keeping seeds off the PML edge and leaving room to flow.
%
% OUTPUT:
%  - vtuStruct : same struct with .density zeroed outside the box, renormalized.
%  - stats     : struct with .NInBox, .NTotal, .InBoxFraction, .SeedableCells.

if nargin < 3 || isempty(marginFrac)
    marginFrac = 0.15;
end

if ~isfield(vtuStruct, 'density')
    error('crop_vessel_to_domain:noDensity', ...
        'vtuStruct.density missing; run apply_velocity_weighted_seeding first.');
end

% Vessel -> image transform: image = R*(vessel - bb) + c
R  = Geometry.Rotation;
bb = Geometry.BoundingBox.Center(:);
c  = Geometry.Center(:);
img = (R * (vtuStruct.points.' - bb) + c).';   % Ncells x 3, [depth width elev]

% Domain box (image frame). Rows: X=depth, Y=lateral, Z=elevation.
D = Geometry.Domain;
box = [D.Xmin, D.Xmax; D.Ymin, D.Ymax; D.Zmin, D.Zmax];

% Shrink each axis by marginFrac of its span.
lo = box(:,1).' + marginFrac/2 * (box(:,2) - box(:,1)).';
hi = box(:,2).' - marginFrac/2 * (box(:,2) - box(:,1)).';

inBox = img(:,1) >= lo(1) & img(:,1) <= hi(1) & ...
        img(:,2) >= lo(2) & img(:,2) <= hi(2) & ...
        img(:,3) >= lo(3) & img(:,3) <= hi(3);

density = vtuStruct.density;
density(~inBox) = 0;

if sum(density) == 0
    error('crop_vessel_to_domain:empty', ...
        ['No seedable vessel cells inside the domain box. Check ' ...
         'Geometry.Domain / startDepth placement, or reduce marginFrac.']);
end

vtuStruct.density = density / sum(density);

stats = struct();
stats.NTotal        = numel(inBox);
stats.NInBox        = nnz(inBox);
stats.InBoxFraction = mean(inBox);
stats.SeedableCells = nnz(vtuStruct.density > 0);

end
