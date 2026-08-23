function value = normalize_settings_types(value)
%NORMALIZE_SETTINGS_TYPES  Recursively convert numeric settings to double.
%
% The GUI and the settings generators can save fields as integer or single
% classes (e.g. Geometry.Rotation). k-Wave arithmetic such as the
% rotation-matrix multiply in generate_streamlines
% ("R_geom' * R_theta_img * R_geom") errors on integer classes
% ("MTIMES is not fully supported for integer classes"). Passing every loaded
% settings struct through this function guarantees all numeric fields are
% double before the simulation runs.
%
% Recurses into nested structs, struct arrays, and cell arrays. Non-numeric
% leaves (char, string, logical, function handles) are left untouched.

if isstruct(value)
    for index = 1:numel(value)
        names = fieldnames(value(index));
        for field_index = 1:numel(names)
            name = names{field_index};
            value(index).(name) = normalize_settings_types(value(index).(name));
        end
    end
elseif iscell(value)
    for index = 1:numel(value)
        value{index} = normalize_settings_types(value{index});
    end
elseif isnumeric(value)
    value = double(value);
end

end
