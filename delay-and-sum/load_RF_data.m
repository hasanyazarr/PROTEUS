function [RF_matrix, FrameNumbers, RFFileNames, PulseInfo] = load_RF_data(resultsFolder,pulsingScheme)
% LOAD_RF_DATA reads RF data files and applies a pulsing scheme.
%
% RF = LOAD_RF_DATA(resultsFolder,pulsingScheme) reads the RF data in the
% folder resultsFolder and applies the pulsing scheme pulsingScheme to the
% data. RF is an Nelem-by-Nt-by-Nframes array, where Nelem is the number of
% transducer elements, Nt the number of time samples, and Nframes the
% number of frames.
%
% [RF, FrameNumbers, RFFileNames, PulseInfo] also returns the original source
% frame numbers, source RF files, and pulse-combination metadata. Export code
% must use FrameNumbers when looking up ground-truth labels.
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

% Total number of frames in the list:
Nframes = length(filelist);

disp('Loading data and applying pulsing scheme')
RF_matrix = zeros(Nelem,Nt,Nframes,class(RF));

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
    
    RF_matrix(:,:,iframe) = RF;
    
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
