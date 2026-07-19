function s = normalize_settings_types(s)
% NORMALIZE_SETTINGS_TYPES  Recursively convert numeric struct fields to double.
%
% The GUI can save settings fields as integer or single classes (e.g.
% Geometry.Rotation). k-Wave arithmetic such as the rotation-matrix multiply
% in generate_streamlines ("R_geom' * R_theta_img * R_geom") errors on integer
% classes ("MTIMES is not fully supported for integer classes"). Passing every
% loaded settings struct through this function guarantees all numeric fields
% are double before the simulation runs.
%
% Non-numeric fields (char, string, cell, logical) and already-double fields
% are left untouched. Nested structs are handled recursively.

if isstruct(s)
    f = fieldnames(s);
    for k = 1:numel(f)
        s.(f{k}) = normalize_settings_types(s.(f{k}));
    end
elseif isnumeric(s) && ~isa(s, 'double')
    s = double(s);
end

end
