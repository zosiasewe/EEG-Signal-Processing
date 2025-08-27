%% Description
%{
Data set covers a project of Tasting 4 different materials
Subjects : 5 participants with Closed nose and 5 participant with opened nose
Materials used are described in the file "Label" as 1,2,3,4

The data from each subjects consists : 160x1280x17 meaning: trials x time x channels

The data was recorded using 512 Hz sampling frequency
Each epoch consists the recorded event of "tasting"

What we want to achieve?
- We want to perform a binary classification on both opened and closed nose
- We want to use ML algorithms for a robust classification of this closed/open nose 0/1 problem

For that we want to divide the problem into 5 subgroups:
    - Test all of the data (160 trials)
    - Test only TM1 ("Test material 1")
    - Test only TM2 ("Test material 2")
    - Test only TM3 ("Test material 3")
    - Test only TM4 ("Test material 4")
%}

%% ---------------------------------------------------------------------------------------------------------------------------
clc
clear
close all

addpath('functions');
addpath('src');

%% Load Data Files

% Load opened nose data
HIO_opened_path = fullfile(pwd, 'opened_nose', 'HIO.mat');
HIO_opened = load(HIO_opened_path);
data_HIO_opened = struct2array(HIO_opened);

KK_opened_path = fullfile(pwd, 'opened_nose', 'KK.mat');
KK_opened = load(KK_opened_path);
data_KK_opened = struct2array(KK_opened);

KUA_opened_path = fullfile(pwd, 'opened_nose', 'KUA.mat');
KUA_opened = load(KUA_opened_path);
data_KUA_opened = struct2array(KUA_opened);

SK_T_opened_path = fullfile(pwd, 'opened_nose', 'SK_T.mat');
SK_T_opened = load(SK_T_opened_path);
data_SK_T_opened = struct2array(SK_T_opened);

SM_opened_path = fullfile(pwd, 'opened_nose', 'SM.mat');
SM_opened = load(SM_opened_path);
data_SM_opened = struct2array(SM_opened);

opened_names = {'HIO', 'KK', 'KUA', 'SK_T', 'SM'};

% Load closed nose data
AC_closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
AC_closed = load(AC_closed_path);
data_AC_closed = struct2array(AC_closed);

BA_closed_path = fullfile(pwd, 'closed_nose', 'BA.mat');
BA_closed = load(BA_closed_path);
data_BA_closed = struct2array(BA_closed);

BO_closed_path = fullfile(pwd, 'closed_nose', 'BO.mat');
BO_closed = load(BO_closed_path);
data_BO_closed = struct2array(BO_closed);

CMY_closed_path = fullfile(pwd, 'closed_nose', 'CMY.mat');
CMY_closed = load(CMY_closed_path);
data_CMY_closed = struct2array(CMY_closed);

JD_closed_path = fullfile(pwd, 'closed_nose', 'JD.mat');
JD_closed = load(JD_closed_path);
data_JD_closed = struct2array(JD_closed);

closed_names = {'AC', 'BA', 'BO', 'CMY', 'JD'};

% Load channel names
channel_names = load("Channel_names.mat");
channels = struct2array(channel_names);

%% Define Variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);                    % 160 trials
n_time = size(data_AC_closed, 2);                      % 1280 time points per trial
n_channels = size(data_AC_closed, 3);                  % 17 channels
trial_duration_sec = n_time / sampling_rate;           % Trial duration in seconds
total_duration_sec = n_trials * trial_duration_sec;    % Total recording duration



%% Combine and Reshape Data

data_closed_combined = cat(4, data_AC_closed, data_BA_closed, data_BO_closed, data_CMY_closed, data_JD_closed);
data_opened_combined = cat(4, data_HIO_opened, data_KK_opened, data_KUA_opened, data_SK_T_opened, data_SM_opened);

% Reshape data for plotting (continuous format)
EEG_data_closed_reshaped = cell(5,1);
EEG_data_opened_reshaped = cell(5,1);

for i = 1:5
    data_closed_permuted = permute(data_closed_combined(:,:,:,i), [3,2,1]);   % [channels × time × trials]
    data_opened_permuted = permute(data_opened_combined(:,:,:,i), [3,2,1]);
    
    EEG_data_closed_reshaped{i} = reshape(data_closed_permuted, n_channels, []); % Continuous format
    EEG_data_opened_reshaped{i} = reshape(data_opened_permuted, n_channels, []);
end

time_axis = linspace(0, total_duration_sec, size(EEG_data_closed_reshaped{1}, 2));
channel_offset = 300;


%----------------------------------------
%% 1. Raw Data 
%----------------------------------------

fprintf(' 1. Raw Data \n\n');
        
fprintf('\n Closed nose subjects:\n');
% Raw Data Plots & PSD plots & SNR calculations
for i = 1:5
    EEG_data_closed = EEG_data_closed_reshaped{i};
    plotContinuousEEG(time_axis, EEG_data_closed, n_channels, channel_offset, closed_names{i}, 'closed', 'raw', channels);
    
    PSD_data_subj = EEG_data_closed_reshaped{i};
    analyzePSD(PSD_data_subj,sampling_rate, n_channels, channels, closed_names{i}, 'closed', 'raw')

end

fprintf('\n Opened nose subjects:\n');
for i = 1:5
    EEG_data_subj_opened = EEG_data_opened_reshaped{i};
    plotContinuousEEG(time_axis, EEG_data_subj_opened, n_channels, channel_offset, opened_names{i}, 'opened', 'raw', channels);
    
    PSD_data_subj_opened = EEG_data_opened_reshaped{i};
    analyzePSD(PSD_data_subj_opened,sampling_rate, n_channels, channels, opened_names{i}, 'opened', 'raw')

end

%% Plot Individual Epochs - Sample Trials
epoch_samples = [40, 50, 60, 70];
epoch_time = (0:n_samples-1) / sampling_rate;

for i = 1:5
    EEG_data_closed = data_closed_combined(:, :, :, i); % all trials for this subject
    plotIndividualEpochs(EEG_data_closed, epoch_samples, epoch_time, closed_names{i}, 'closed', n_channels, channel_offset, channels, 'raw');
    
    EEG_data_opened = data_closed_combined(:, :, :, i); % all trials for this subject
    plotIndividualEpochs(EEG_data_opened, epoch_samples, epoch_time, opened_names{i}, 'opened', n_channels, channel_offset, channels, 'raw');
end


%----------------------------------------
%% 2. Filtering
%----------------------------------------

fprintf('\n  2. Filtering \n\n');

% Filter parameters
low_cutoff = 0.5;
high_cutoff = 30;
filter_order = 2;

EEG_filtered_closed = zeros(n_channels, n_samples, n_trials, 5);
EEG_filtered_opened = zeros(n_channels, n_samples, n_trials, 5);
    
for i = 1:5
    EEG_filtering_closed = EEG_data_closed_reshaped{i};  % [channels x time]
    EEG_filtered_closed(:,:,:,i) = reshape(filterEEG(EEG_filtering_closed, low_cutoff, high_cutoff, filter_order, n_channels, sampling_rate), n_channels, n_samples, n_trials);

    EEG_filtering_opened = EEG_data_opened_reshaped{i};
    EEG_filtered_opened(:,:,:,i) = reshape(filterEEG(EEG_filtering_opened, low_cutoff, high_cutoff, filter_order, n_channels, sampling_rate), n_channels, n_samples, n_trials);
end



%% Plot Filtered Data

% Continuous plot & PSD plot & SNR
for i = 1:5
    Closed_filtered_continuous = reshape(EEG_filtered_closed(:,:,:,i), n_channels, []);
    plotContinuousEEG(time_axis, Closed_filtered_continuous, n_channels, channel_offset, closed_names{i}, 'closed', 'filt', channels);
    analyzePSD(Closed_filtered_continuous, sampling_rate, n_channels, channels, closed_names{i}, 'closed', 'filt');
end

for i = 1:5
    Opened_filtered_continuous = reshape(EEG_filtered_opened(:,:,:,i), n_channels, []);
    plotContinuousEEG(time_axis, Opened_filtered_continuous, n_channels, channel_offset, opened_names{i}, 'opened', 'filt', channels);
    analyzePSD(Opened_filtered_continuous, sampling_rate, n_channels, channels, opened_names{i}, 'opened', 'filt');
end

%% Plot Filtered Epochs

epoch_samples = [40, 50, 60, 70];

for i = 1:5
    Epoch_closed_data = zeros(n_channels, n_samples, length(epoch_samples));
    for t_idx = 1:length(epoch_samples)
        trial_num = epoch_samples(t_idx);
        Epoch_closed_data(:, :, t_idx) = EEG_filtered_closed(:, :, trial_num, i);
    end
    plotIndividualEpochs(Epoch_closed_data, epoch_samples, epoch_time, closed_names{i}, 'closed', n_channels, channel_offset, channels, 'filt');
end


for i = 1:5
    Epoch_opened_data = zeros(n_channels, n_samples, length(epoch_samples));
    for t_idx = 1:length(epoch_samples)
        trial_num = epoch_samples(t_idx);
        Epoch_opened_data(:, :, t_idx) = EEG_filtered_opened(:, :, trial_num, i);
    end
    plotIndividualEpochs(Epoch_opened_data, epoch_samples, epoch_time, opened_names{i}, 'opened', n_channels, channel_offset, channels, 'filt');
end

%----------------------------------------
%% 3. Artifact removal - ICA
%----------------------------------------

fprintf('\n   3. ICA \n\n');
n_components = 4;
max_iterations = 1000;
tolerance = 1e-6;
learning_rate = 1.0;

% ICA for Closed nose subjects
clean_eeg_closed = zeros(size(EEG_filtered_closed));
for i = 1:5
    fprintf('\nICA on Closed Nose Subject: %s\n', closed_names{i});
    clean_eeg_closed(:,:,:,i) = runICASubject(EEG_filtered_closed(:,:,:,i), 4, sampling_rate, closed_names{i}, 'closed', epoch_time);
end

% ICA for Opened nose subjects
clean_eeg_opened = zeros(size(EEG_filtered_opened));
for i = 1:5
    fprintf('\nICA on Opened Nose Subject: %s\n', opened_names{i});
    clean_eeg_opened(:,:,:,i) = runICASubject(EEG_filtered_opened(:,:,:,i), 4, sampling_rate, opened_names{i}, 'opened', epoch_time);
end

%----------------------------------------
%% 4. Feature Extraction
%----------------------------------------

bands = [0.5 4; 4 8; 8 13; 13 30; 30 100]; % freq bands
n_trials = size(data_AC_closed, 1); % 160 trials
sampling_rate = 512;

all_features_closed = [];
all_features_opened = [];

% Features for Closed nose subjects
fprintf('\nMaking features for Closed nose subjects...\n');
for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_closed(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_closed = [all_features_closed; subj_features];
end

% Features for Opened nose subjects
fprintf('\nMaking features for Opened nose subjects...\n');
for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_opened(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_opened = [all_features_opened; subj_features];
end

all_features_combined = [all_features_closed; all_features_opened];
feature_labels = [zeros(size(all_features_closed, 1), 1); ones(size(all_features_opened, 1), 1)]; % 0=closed, 1=opened

% z-score normalization
feature_matrix_normalized = (all_features_combined - mean(all_features_combined)) ./ std(all_features_combined);

