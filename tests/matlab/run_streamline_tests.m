% Runner for the streamline-module MATLAB behaviour tests.
%   matlab -batch "run_streamline_tests"
here = fileparts(mfilename('fullpath'));
tests = {'test_crop_vessel_to_slab', 'test_validate_tile_placement'};
for k = 1:numel(tests)
    fprintf('--- %s\n', tests{k});
    run(fullfile(here, [tests{k} '.m']));
end
disp('streamline tests: OK');
