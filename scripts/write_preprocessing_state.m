function statePath = write_preprocessing_state(vizOut, PreprocessingState)
%WRITE_PREPROCESSING_STATE Record how a viz pass preprocessed the run.
%
% process_run builds the image ROI, the RF sample window, the SVD fit frames
% and the intensity normalization reference, then used to drop all of it: the
% only trace was stdout, so asserting on the crop of a finished run meant
% scraping a notebook log.
%
% Written under vizOut rather than beside the RF data on purpose. One results
% folder can be processed several times with different PREPROCESSING_OPTIONS,
% each into its own vizOut; the state describes that pass, not the RF, so
% putting it with the RF would have each pass overwrite the last.
%
% JSON because the consumers are the Python scripts in scripts/analysis.

if ~exist(vizOut, 'dir')
    mkdir(vizOut);
end

statePath = fullfile(vizOut, 'preprocessing_state.json');

% NormalizationReference is a per-frame vector under 'per_frame' and a scalar
% otherwise. Force a row so the JSON shape does not depend on the mode.
if isfield(PreprocessingState, 'NormalizationReference')
    PreprocessingState.NormalizationReference = ...
        reshape(double(PreprocessingState.NormalizationReference), 1, []);
end
if isfield(PreprocessingState, 'SVDFitFrameNumbers')
    PreprocessingState.SVDFitFrameNumbers = ...
        reshape(double(PreprocessingState.SVDFitFrameNumbers), 1, []);
end

State = PreprocessingState;
State.WrittenAt = datestr(now, 'yyyy-mm-ddTHH:MM:SS');

try
    text = jsonencode(State, 'PrettyPrint', true);
catch err
    % A field that will not encode must not cost the whole record.
    warning('write_preprocessing_state:EncodeFailed', ...
        'Falling back to a reduced record: %s', err.message);
    Reduced = struct();
    for field = {'ImageROI', 'SampleRange', 'SampleRangeSource', ...
            'SampleRangeMargin', 'SplitID', 'SplitMode', 'SVDCutoff', ...
            'SVDFitFrameNumbers', 'SVDFitScope', 'NormalizationMode'}
        if isfield(State, field{1})
            Reduced.(field{1}) = State.(field{1});
        end
    end
    Reduced.EncodeError = err.message;
    text = jsonencode(Reduced, 'PrettyPrint', true);
end

fid = fopen(statePath, 'w');
if fid == -1
    error('write_preprocessing_state:CannotWrite', ...
        'Could not open %s for writing.', statePath);
end
closeFile = onCleanup(@() fclose(fid));
fprintf(fid, '%s', text);

fprintf('  preprocessing_state.json\n');

end
