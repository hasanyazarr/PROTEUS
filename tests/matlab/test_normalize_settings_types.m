repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(repoRoot);

%% Integer and single numeric fields become double.
s.a = int32(5);
s.b = single(2.5);
s.c = 7;
out = normalize_settings_types(s);
assert(isa(out.a, 'double') && out.a == 5);
assert(isa(out.b, 'double') && out.b == 2.5);
assert(isa(out.c, 'double') && out.c == 7);

%% Nested structs are converted recursively.
nested.Geometry.Rotation = int16([1 0 0; 0 1 0; 0 0 1]);
nested.Geometry.Domain.Xmin = single(-6e-3);
out = normalize_settings_types(nested);
assert(isa(out.Geometry.Rotation, 'double'));
assert(isa(out.Geometry.Domain.Xmin, 'double'));
% The rotation-matrix multiply that motivated this function must now work.
R = out.Geometry.Rotation;
assert(isequal(R' * R, eye(3)));

%% Struct arrays are converted in every element, not just the first.
arr(1).value = int8(1);
arr(2).value = int8(2);
arr(3).value = single(3);
out = normalize_settings_types(arr);
assert(numel(out) == 3);
for k = 1:3
    assert(isa(out(k).value, 'double'));
end

%% Cell arrays are converted element by element.
c = {int32(1), {single(2)}, struct('inner', int16(3))};
out = normalize_settings_types(c);
assert(isa(out{1}, 'double'));
assert(isa(out{2}{1}, 'double'));
assert(isa(out{3}.inner, 'double'));

%% Non-numeric leaves survive untouched.
mixed.name = 'renal_tree';
mixed.flag = true;
mixed.empty = [];
mixed.handle = @sin;
out = normalize_settings_types(mixed);
assert(ischar(out.name) && strcmp(out.name, 'renal_tree'));
assert(islogical(out.flag) && out.flag);
assert(isempty(out.empty));
assert(isa(out.handle, 'function_handle'));

disp('test_normalize_settings_types: all assertions passed');
