function [RF_matrix, FrameNumbers, RFFileNames, PulseInfo, SampleRange] = ...
    load_RF_data(resultsFolder,pulsingScheme,sampleRange)
% LOAD_RF_DATA reads RF data files and applies a pulsing scheme.
%
% RF = LOAD_RF_DATA(resultsFolder,pulsingScheme) reads the RF data in the
% folder resultsFolder and applies the pulsing scheme pulsingScheme to the
% data. RF is an Nelem-by-Nt-by-Nframes array, where Nelem is the number of
% transducer elements, Nt the number of time samples, and Nframes the
% number of frames.
%
% RF = LOAD_RF_DATA(...,sampleRange) keeps only samples
% sampleRange(1):sampleRange(2) of each trace. The caller is expected to
% derive that window from the reconstruction it is going to perform: a
% beamformer reads only the samples its shallowest and deepest pixels need,
% and on the v7 acquisition that is about a third of each trace. Cropping
% here rather than after the load is what saves the memory. Pass [] to keep
% every sample.
%
% [RF, FrameNumbers, RFFileNames, PulseInfo, SampleRange] also returns the
% original source frame numbers, source RF files, pulse-combination metadata,
% and the sample window actually applied after clamping to the trace length.
% Export code must use FrameNumbers when looking up ground-truth labels.
%
% Guillaume Lajoinie, Nathan Blanken, University of Twente, 2023

% Get a list of all the frames in the results folder:
filelist = dir(fullfile(resultsFolder,'Frame*.mat'));

% Get the frame numbers of the files in the list:
FrameNumbers = arrayfun(@(F) str2double(F.name(7:end-4)),filelist);

% Sort the file list by frame number:
[~, I] = sort(FrameNumbers);
filelist = filelist(I);
FrameNumbers = FrameNumbers(I);
RFFileNames = arrayfun(@(F) fullfile(F.folder, F.name), filelist, ...
    'UniformOutput', false);
PulseInfo = get_pulse_info(pulsingScheme);

% Load a sample RF data frame:
load(fullfile(filelist(1).folder, filelist(1).name),'RF');
RF = RF{1};

% RF data properties:
Nt = size(RF,2);    % Number of samples per RF line
Nelem = size(RF,1); % Number of transducer elements

% Sample window to keep:
if nargin < 3 || isempty(sampleRange)
    SampleRange = [1 Nt];
else
    if numel(sampleRange) ~= 2 || any(~isfinite(sampleRange)) || ...
            sampleRange(2) < sampleRange(1) || sampleRange(1) < 1
        error('load_RF_data:InvalidSampleRange', ...
            'sampleRange must be [first last] with 1 <= first <= last.');
    end
    SampleRange = [max(1, round(sampleRange(1))), ...
                   min(Nt, round(sampleRange(2)))];
end
Nkeep = SampleRange(2) - SampleRange(1) + 1;
if SampleRange(2) < Nt || SampleRange(1) > 1
    fprintf(['  keeping samples %d-%d of %d ' ...
        '(%.1f%% of each trace)\n'], SampleRange(1), SampleRange(2), Nt, ...
        100*Nkeep/Nt);
end

% Total number of frames in the list:
Nframes = length(filelist);

disp('Loading data and applying pulsing scheme')
RF_matrix = zeros(Nelem,Nkeep,Nframes,class(RF));

for iframe = 1:Nframes
    
    load(fullfile(filelist(iframe).folder, filelist(iframe).name),'RF');
    
    switch pulsingScheme
        case 'Amplitude modulation'
            RF = RF{3}-RF{1}-RF{2};
        case 'Pulse inversion'
            RF = RF{1}+RF{2};
        case 'Amplitude modulation with pulse inversion'
            RF = RF{3}+RF{1}+RF{2};
        case 'Standard'
            RF = RF{1};
            
    end
    
    RF_matrix(:,:,iframe) = RF(:,SampleRange(1):SampleRange(2));
    
end

end


function PulseInfo = get_pulse_info(pulsingScheme)
PulseInfo.PulsingScheme = pulsingScheme;
switch pulsingScheme
    case 'Amplitude modulation'
        PulseInfo.PulseIDsUsed = [3 1 2];
        PulseInfo.CombinationFormula = 'RF{3}-RF{1}-RF{2}';
        PulseInfo.LabelPulsePolicy = ...
            'pulse_resolved_labels_plus_combined_target_from_pulses_3_1_2';
    case 'Pulse inversion'
        PulseInfo.PulseIDsUsed = [1 2];
        PulseInfo.CombinationFormula = 'RF{1}+RF{2}';
        PulseInfo.LabelPulsePolicy = ...
            'pulse_resolved_labels_plus_combined_target_from_pulses_1_2';
    case 'Amplitude modulation with pulse inversion'
        PulseInfo.PulseIDsUsed = [3 1 2];
        PulseInfo.CombinationFormula = 'RF{3}+RF{1}+RF{2}';
        PulseInfo.LabelPulsePolicy = ...
            'pulse_resolved_labels_plus_combined_target_from_pulses_3_1_2';
    case 'Standard'
        PulseInfo.PulseIDsUsed = 1;
        PulseInfo.CombinationFormula = 'RF{1}';
        PulseInfo.LabelPulsePolicy = 'single_pulse1_label';
    otherwise
        error('load_RF_data:UnknownPulsingScheme', ...
            'Unknown pulsing scheme: %s', pulsingScheme);
end
end
