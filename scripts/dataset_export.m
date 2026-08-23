function dataset_export(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, OUT_FOLDER, ...
    SIGMA_PX, ELEVATION_FILTER_MM, PREPROCESSING_OPTIONS)
% DATASET_EXPORT  Deprecated compatibility wrapper for legacy SR export.
%
% Active post-processing no longer creates super-resolution datasets. New
% research code should consume RF, ground truth, and Casorati artifacts
% directly. This wrapper preserves reproducibility of historical SR runs.

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

warning('dataset_export:Deprecated', ...
    ['dataset_export is legacy and is not part of the active research ', ...
     'pipeline. Calling legacy.sr.process_run for compatibility.']);

legacy.sr.process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER, '', OUT_FOLDER, ...
    'preview', 60, SIGMA_PX, ELEVATION_FILTER_MM, PREPROCESSING_OPTIONS);

end
