function [medium, vessel, MediumMetadata] = define_medium(...
    Grid, Medium, Geometry, FlowSimulationParameters)
%DEFINE_MEDIUM returns a k-Wave medium struct (a voxel-based representation
%of the acoustic properties of the medium) based on the grid properties in
%Grid and the medium properties in Medium.
%
% The mesh representation of the vessel geometry (the STL file specified in
% Geometry) is converted to a voxel-based representation and embedded in
% the voxel-based medium.
%
% [medium, vessel] = DEFINE_MEDIUM(Grid, Medium, Geometry) returns the
% variables medium and vessel.
%
% medium is a struct with fields:
% - sound_speed
% - density
% - BonA
% - alpha_coeff
% - alpha_power
% - alpha_mode (optional)
%
% vessel is a logical mask with the dimensions of the grid representing the
% location of the vessel.
%
% See for more information on the k-Wave medium:
% http://www.k-wave.org/documentation/example_ivp_heterogeneous_medium.php
%
% See also define_grid.m.
%
% Nathan Blanken, University of Twente, 2023

% Grid dimensions
Nx = Grid.Nx; Ny = Grid.Ny; Nz = Grid.Nz;
if nargin < 4
    FlowSimulationParameters = struct();
end
MediumMetadata.Tiling = get_medium_tiling_metadata(FlowSimulationParameters);

%==========================================================================
% Medium inhomegeneity
%==========================================================================

% Normal distribution truncated at +/-cutoff standard deviations:
cutoff = Medium.InhomogeneityCutoff;
pd = truncate(makedist('Normal'),-cutoff,cutoff);

% Make the speed of sound and density inhomogeneous to generate a linear
% scatterer background:
background_map   = 1 + Medium.Inhomogeneity*...
                   random(pd,[Nx, Ny, Nz]);
               
%==========================================================================
% Background tissue maps
%==========================================================================
medium.sound_speed = Medium.SpeedOfSound * background_map;  %[m/s]
medium.density     = Medium.Density      * background_map;  %[kg/m^3]

medium.BonA        = Medium.BonA         * ones(Nx,Ny,Nz);

% power law absorption prefactor [dB/(MHz^y cm)]:
medium.alpha_coeff = Medium.AttenuationA * ones(Nx,Ny,Nz);
medium.alpha_power = Medium.AttenuationB;

%==========================================================================
% Dispersion
%==========================================================================
% The dispersion term in k-Wave is derived via the Kramers-Kronig
% relations. There exists a singularity for alpha_power = 1 (see page 33 of
% the k-Wave manual, Manual Version 1.1,  for further details). Do not
% simulate dispersion for powers close to 1.
if abs(medium.alpha_power - 1) < 0.1
    medium.alpha_mode =  'no_dispersion';
end

%==========================================================================
% Vessel
%==========================================================================

% Location of the STL file:
Geometry.STLfile = [Geometry.GeometriesPath filesep Geometry.Folder ...
    filesep Geometry.STLfile];

if Geometry.EmbedVessel
    % Read STL data:
    V_raw = READ_stl(Geometry.STLfile);

    vessel = zeros(Nx, Ny, Nz, 'logical');
    transforms = MediumMetadata.Tiling.Transforms;
    for tileIdx = 1:numel(transforms)
        V = apply_tile_transform_to_stl(V_raw, transforms(tileIdx), Geometry);
        vessel = vessel | VOXELISE(Grid.x, Grid.y, Grid.z, V);
    end
else
    vessel = zeros(Nx, Ny, Nz,'logical');
end

% Assign the tissue properties of the vessel:
medium.sound_speed(vessel) = Medium.Vessel.SpeedOfSound;
medium.density(vessel)     = Medium.Vessel.Density;
medium.BonA(vessel)        = Medium.Vessel.BonA; 
    
end


%==========================================================================
% FUNCTIONS
%==========================================================================

function meshXYZ = rotate_stl(meshXYZ,R)
%ROTATE_STL rotates STL coordinates meshXYZ with rotation matrix R.
%
% The N-by-3-by-3 array meshXYZ is defined as the output of READ_stl.m in
% the mesh voxelisation toolbox by Adam H. Aitkenhead:
%  1 row for each facet
%  3 cols for the x,y,z coordinates
%  3 pages for the three vertices
%
% The 3-by-3 rotation matrix is defined such that B = R*A, where A is a
% column vector and B is the rotated column vector.

for i = 1:3
    V = transpose(meshXYZ(:,:,i)); % 3-by-N matrix of vertex coordinates
    V = R*V;                       % Rotate the vertex coordinates
    meshXYZ(:,:,i) = transpose(V); % Assign result
end


function V = apply_tile_transform_to_stl(V_raw, transform, Geometry)
V = V_raw * Geometry.STLunit;
BB = reshape(Geometry.BoundingBox.Center, 3, 1);
for i = 1:3
    vertices = transpose(V(:,:,i));
    vertices = transform.Rotation * (vertices - BB) + BB + transform.Offset;
    V(:,:,i) = transpose(vertices);
end
V = V - transpose(Geometry.BoundingBox.Center);
V = rotate_stl(V,Geometry.Rotation);
V = V + transpose(Geometry.Center);
end


function tiling = get_medium_tiling_metadata(FlowSimulationParameters)
tiling.Enabled = false;
tiling.TransformFrame = 'vessel_to_image_consistent';
tiling.Transforms = struct( ...
    'TileID', 1, ...
    'Rotation', eye(3), ...
    'Offset', zeros(3, 1), ...
    'TransformFrame', tiling.TransformFrame);
if isfield(FlowSimulationParameters, 'Tiling')
    tiling = FlowSimulationParameters.Tiling;
    if ~isfield(tiling, 'TransformFrame')
        tiling.TransformFrame = 'vessel_to_image_consistent';
    end
    if ~isfield(tiling, 'Transforms') || isempty(tiling.Transforms)
        tiling.Transforms = struct( ...
            'TileID', 1, ...
            'Rotation', eye(3), ...
            'Offset', zeros(3, 1), ...
            'TransformFrame', tiling.TransformFrame);
    end
end
end

end
