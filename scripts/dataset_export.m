function dataset_export(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER, ...
    SIGMA_PX, ELEVATION_FILTER_MM, PREPROCESSING_OPTIONS)
% DATASET_EXPORT  Compatibility wrapper for SR dataset export.
%
% The export implementation lives in process_run.m so RF loading, SVD
% preprocessing, frame identity, pulse-aware labels, visibility metadata, and
% overlap-safe targets have one shared contract.

if nargin < 5 || isempty(SIGMA_PX)
    SIGMA_PX = 1.5;
end
if nargin < 6 || isempty(ELEVATION_FILTER_MM)
    ELEVATION_FILTER_MM = 1.0;
end
if nargin < 7 || isempty(PREPROCESSING_OPTIONS)
    error('dataset_export:MissingPreprocessingOptions', ...
        'Dataset export requires PREPROCESSING_OPTIONS with SplitMode and SVD policy.');
end

process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, '', OUT_FOLDER, ...
    'preview', 60, SIGMA_PX, ELEVATION_FILTER_MM, PREPROCESSING_OPTIONS);

end
