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

fprintf(' EEG Data Processing for Taste Experiment \n');

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
fprintf('\nData specifications:\n');
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);                    % 160 trials
n_time = size(data_AC_closed, 2);                      % 1280 time points per trial
n_channels = size(data_AC_closed, 3);                  % 17 channels
trial_duration_sec = n_time / sampling_rate;           % Trial duration in seconds
total_duration_sec = n_trials * trial_duration_sec;    % Total recording duration

fprintf('  - Sampling rate: %d Hz\n', sampling_rate);
fprintf('  - Number of trials: %d\n', n_trials);
fprintf('  - Samples per trial: %d\n', n_samples);
fprintf('  - Number of channels: %d\n', n_channels);
fprintf('  - Trial duration: %.2f seconds\n', trial_duration_sec);
fprintf('  - Total duration: %.2f seconds\n\n', total_duration_sec);

%% Combine and Reshape Data
]
% Combine data from all subjects
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
channel_offset = 300; % Separation between channels in plots


%----------------------------------------
%% 1. Raw Data 
%----------------------------------------

fprintf(' 1. Raw Data \n\n');
        
% Plot Raw EEG Data - All Subjects
% All closed nose subjects first
fprintf('\nProcessing CLOSED nose subjects:\n');
for i = 1:5
    
    % Continuous EEG plot
    figure;
    hold on;
    for ch = 1:n_channels
        plot(time_axis, EEG_data_closed_reshaped{i}(ch, :) + (ch - 1) * channel_offset);
    end
    title('Continuous Raw EEG Data - All 160 Trials');
    subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * channel_offset);
    yticklabels(channels(1:n_channels));
    ylim([-channel_offset, channel_offset * n_channels]);
    xlim([0, total_duration_sec]);
    grid on;
    hold off;
    
    % Power Spectral Density analysis
    figure;
    
    snr_values = zeros(1, n_channels);
    signal_band = [0.5 30];
    noise_band = [45 100];
    all_psd_data = [];
    
    for ch = 1:n_channels
        [psd_values, freq_axis] = pwelch(EEG_data_closed_reshaped{i}(ch, :), [], [], [], sampling_rate);
        all_psd_data = [all_psd_data, psd_values];
        
        signal_power = sum(psd_values(freq_axis >= signal_band(1) & freq_axis <= signal_band(2)));
        noise_power = sum(psd_values(freq_axis >= noise_band(1) & freq_axis <= noise_band(2)));
        snr_values(ch) = 10*log10(signal_power/noise_power);
    end
    
    % Plot average PSD
    avg_psd = mean(all_psd_data, 2);
    avg_psd_db = 10*log10(avg_psd);
    
    plot(freq_axis, avg_psd_db, 'b-', 'LineWidth', 1.2, 'Color', "#7E2F8E");
    title('Average Power Spectral Density');
    subtitle(sprintf('Subject: %s (Closed Nose) - Raw Data', closed_names{i}));
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    xlim([0 256]);
    ylim([-80 40]);
    grid on;
    
    % Display SNR values
    fprintf('    SNR values for %s (closed nose):\n', closed_names{i});
    for ch = 1:n_channels
        fprintf('      %s: %.2f dB\n', channels{ch}, snr_values(ch));
    end
end

% Process all opened nose subjects
fprintf('\nProcessing OPENED nose subjects:\n');
for i = 1:5
    fprintf('  Processing subject %s (opened nose)...\n', opened_names{i});
    
    % Continuous EEG plot
    figure;
    hold on;
    for ch = 1:n_channels
        plot(time_axis, EEG_data_opened_reshaped{i}(ch, :) + (ch - 1) * channel_offset);
    end
    title('Continuous Raw EEG Data - All 160 Trials');
    subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * channel_offset);
    yticklabels(channels(1:n_channels));
    ylim([-channel_offset, channel_offset * n_channels]);
    xlim([0, total_duration_sec]);
    grid on;
    hold off;
    
    % Power Spectral Density analysis
    figure;
    fprintf('    Computing PSD and SNR for %s...\n', opened_names{i});
    
    snr_values = zeros(1, n_channels);
    signal_band = [0.5 30];
    noise_band = [45 100];
    all_psd_data = [];
    
    for ch = 1:n_channels
        [psd_values, freq_axis] = pwelch(EEG_data_opened_reshaped{i}(ch, :), [], [], [], sampling_rate);
        all_psd_data = [all_psd_data, psd_values];
        
        signal_power = sum(psd_values(freq_axis >= signal_band(1) & freq_axis <= signal_band(2)));
        noise_power = sum(psd_values(freq_axis >= noise_band(1) & freq_axis <= noise_band(2)));
        snr_values(ch) = 10*log10(signal_power/noise_power);
    end
    
    % Plot average PSD
    avg_psd = mean(all_psd_data, 2);
    avg_psd_db = 10*log10(avg_psd);
    
    plot(freq_axis, avg_psd_db, 'b-', 'LineWidth', 1.2, 'Color', "#7E2F8E");
    title('Average Power Spectral Density');
    subtitle(sprintf('Subject: %s (Opened Nose) - Raw Data', opened_names{i}));
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    xlim([0 256]);
    ylim([-80 40]);
    grid on;
    
    % Display SNR values
    fprintf('    SNR values for %s (opened nose):\n', opened_names{i});
    for ch = 1:n_channels
        fprintf('      %s: %.2f dB\n', channels{ch}, snr_values(ch));
    end
end

%% Plot Individual Epochs - Sample Trials
fprintf('\nGenerating epoch plots for sample trials...\n');
epoch_samples = [40, 50, 60, 70];
epoch_time = (0:n_samples-1) / sampling_rate;

% Plot epochs for closed nose subjects
for i = 1:5
    
    figure;
    for trial_idx = 1:length(epoch_samples)
        subplot(2, 2, trial_idx);
        hold on;
        
        for ch = 1:n_channels
            plot(epoch_time, data_closed_combined(epoch_samples(trial_idx), :, ch, i) + (ch - 1) * channel_offset);
        end
        
        title(sprintf('Trial %d - Raw EEG', epoch_samples(trial_idx)));
        subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels - 1) * channel_offset);
        yticklabels(channels(1:n_channels));
        ylim([-channel_offset, channel_offset * n_channels]);
        xlim([0, 2.5]);
        grid on;
        hold off;
    end
end

% Plot epochs for opened nose subjects
for i = 1:5
    
    figure;
    for trial_idx = 1:length(epoch_samples)
        subplot(2, 2, trial_idx);
        hold on;
        
        for ch = 1:n_channels
            plot(epoch_time, data_opened_combined(epoch_samples(trial_idx), :, ch, i) + (ch - 1) * channel_offset);
        end
        
        title(sprintf('Trial %d - Raw EEG', epoch_samples(trial_idx)));
        subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels - 1) * channel_offset);
        yticklabels(channels(1:n_channels));
        ylim([-channel_offset, channel_offset * n_channels]);
        xlim([0, 2.5]);
        grid on;
        hold off;
    end
end

%----------------------------------------
%% 2. Filtering
%----------------------------------------

fprintf('\n  2. Filtering \n\n');
fprintf('Applying bandpass filter (0.5-30 Hz) to all subjects...\n');

% Filter parameters
low_cutoff = 0.5;
high_cutoff = 30;
filter_order = 2;

% Initialize filtered data storage
EEG_filtered_closed = zeros(n_channels, n_samples, n_trials, 5);
EEG_filtered_opened = zeros(n_channels, n_samples, n_trials, 5);

% Filter closed nose data
fprintf('\nFiltering CLOSED nose data:\n');
for i = 1:5
    fprintf('  Filtering subject %s (closed nose)...\n', closed_names{i});
    
    nyquist_freq = sampling_rate / 2;
    [b_highpass, a_highpass] = butter(filter_order, low_cutoff / nyquist_freq, 'high');
    [b_lowpass, a_lowpass] = butter(filter_order, high_cutoff / nyquist_freq, 'low');
    
    filtered_signal = zeros(size(EEG_data_closed_reshaped{i}));
    
    for ch = 1:n_channels
        highpass_signal = filtfilt(b_highpass, a_highpass, EEG_data_closed_reshaped{i}(ch, :));
        filtered_signal(ch, :) = filtfilt(b_lowpass, a_lowpass, highpass_signal);
    end
    
    EEG_filtered_closed(:,:,:,i) = reshape(filtered_signal, n_channels, n_samples, n_trials);
end

% Filter opened nose data
fprintf('\nFiltering OPENED nose data:\n');
for i = 1:5
    fprintf('  Filtering subject %s (opened nose)...\n', opened_names{i});
    
    nyquist_freq = sampling_rate / 2;
    [b_highpass, a_highpass] = butter(filter_order, low_cutoff / nyquist_freq, 'high');
    [b_lowpass, a_lowpass] = butter(filter_order, high_cutoff / nyquist_freq, 'low');
    
    filtered_signal = zeros(size(EEG_data_opened_reshaped{i}));
    
    for ch = 1:n_channels
        highpass_signal = filtfilt(b_highpass, a_highpass, EEG_data_opened_reshaped{i}(ch, :));
        filtered_signal(ch, :) = filtfilt(b_lowpass, a_lowpass, highpass_signal);
    end
    
    EEG_filtered_opened(:,:,:,i) = reshape(filtered_signal, n_channels, n_samples, n_trials);
end

%% Plot Filtered Data

% Plot filtered continuous data for closed nose
for i = 1:5
    
    filtered_continuous = reshape(EEG_filtered_closed(:,:,:,i), n_channels, []);
    
    % Continuous plot
    figure;
    hold on;
    for ch = 1:n_channels
        plot(time_axis, filtered_continuous(ch, :) + (ch - 1) * channel_offset);
    end
    title('Continuous Filtered EEG Data (0.5-30 Hz)');
    subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * channel_offset);
    yticklabels(channels(1:n_channels));
    ylim([-channel_offset, channel_offset * n_channels]);
    xlim([0, total_duration_sec]);
    grid on;
    hold off;
    
    % PSD analysis for filtered data
    figure;
    
    snr_filtered = zeros(1, n_channels);
    all_psd_filtered = [];
    
    for ch = 1:n_channels
        [psd_values, freq_axis] = pwelch(filtered_continuous(ch, :), [], [], [], sampling_rate);
        all_psd_filtered = [all_psd_filtered, psd_values];
        
        signal_power = sum(psd_values(freq_axis >= 0.5 & freq_axis <= 30));
        noise_power = sum(psd_values(freq_axis >= 45 & freq_axis <= 100));
        snr_filtered(ch) = 10*log10(signal_power/noise_power);
    end
    
    avg_psd_filtered = mean(all_psd_filtered, 2);
    avg_psd_filtered_db = 10*log10(avg_psd_filtered);
    
    plot(freq_axis, avg_psd_filtered_db, 'b-', 'LineWidth', 1.2, 'Color', "#7E2F8E");
    title('Average Power Spectral Density - Filtered Data');
    subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    xlim([0 256]);
    ylim([-80 40]);
    grid on;
    
    % Display filtered SNR values
    fprintf('    Filtered SNR values for %s (closed nose):\n', closed_names{i});
    for ch = 1:n_channels
        fprintf('      %s: %.2f dB\n', channels{ch}, snr_filtered(ch));
    end
end

% Plot filtered continuous data for opened nose
for i = 1:5
    
    filtered_continuous = reshape(EEG_filtered_opened(:,:,:,i), n_channels, []);
    
    % Continuous plot
    figure;
    hold on;
    for ch = 1:n_channels
        plot(time_axis, filtered_continuous(ch, :) + (ch - 1) * channel_offset);
    end
    title('Continuous Filtered EEG Data (0.5-30 Hz)');
    subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * channel_offset);
    yticklabels(channels(1:n_channels));
    ylim([-channel_offset, channel_offset * n_channels]);
    xlim([0, total_duration_sec]);
    grid on;
    hold off;
    
    % PSD analysis for filtered data
    figure;
    
    snr_filtered = zeros(1, n_channels);
    all_psd_filtered = [];
    
    for ch = 1:n_channels
        [psd_values, freq_axis] = pwelch(filtered_continuous(ch, :), [], [], [], sampling_rate);
        all_psd_filtered = [all_psd_filtered, psd_values];
        
        signal_power = sum(psd_values(freq_axis >= 0.5 & freq_axis <= 30));
        noise_power = sum(psd_values(freq_axis >= 45 & freq_axis <= 100));
        snr_filtered(ch) = 10*log10(signal_power/noise_power);
    end
    
    avg_psd_filtered = mean(all_psd_filtered, 2);
    avg_psd_filtered_db = 10*log10(avg_psd_filtered);
    
    plot(freq_axis, avg_psd_filtered_db, 'b-', 'LineWidth', 1.2, 'Color', "#7E2F8E");
    title('Average Power Spectral Density - Filtered Data');
    subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    xlim([0 256]);
    ylim([-80 40]);
    grid on;
    
    % Display filtered SNR values
    fprintf('    Filtered SNR values for %s (opened nose):\n', opened_names{i});
    for ch = 1:n_channels
        fprintf('      %s: %.2f dB\n', channels{ch}, snr_filtered(ch));
    end
end

%% Plot Filtered Epochs

% Plot filtered epochs for closed nose
for i = 1:5    
    figure;
    for trial_idx = 1:length(epoch_samples)
        subplot(2, 2, trial_idx);
        hold on;
        
        for ch = 1:n_channels
            plot(epoch_time, EEG_filtered_closed(ch, :, epoch_samples(trial_idx), i) + (ch - 1) * channel_offset);
        end
        
        title(sprintf('Trial %d - Filtered EEG', epoch_samples(trial_idx)));
        subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels - 1) * channel_offset);
        yticklabels(channels(1:n_channels));
        ylim([-channel_offset, channel_offset * n_channels]);
        xlim([0, 2.5]);
        grid on;
        hold off;
    end
end

% Plot filtered epochs for opened nose
for i = 1:5    
    figure;
    for trial_idx = 1:length(epoch_samples)
        subplot(2, 2, trial_idx);
        hold on;
        
        for ch = 1:n_channels
            plot(epoch_time, EEG_filtered_opened(ch, :, epoch_samples(trial_idx), i) + (ch - 1) * channel_offset);
        end
        
        title(sprintf('Trial %d - Filtered EEG', epoch_samples(trial_idx)));
        subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels - 1) * channel_offset);
        yticklabels(channels(1:n_channels));
        ylim([-channel_offset, channel_offset * n_channels]);
        xlim([0, 2.5]);
        grid on;
        hold off;
    end
end

%----------------------------------------
%% 3. Artifact removal - ICA
%----------------------------------------

fprintf('\n   3. Artifact Removal (ICA) \n\n');

n_components = 4;
max_iterations = 1000;
tolerance = 1e-6;
learning_rate = 1.0;

%% ICA for CLOSED nose subjects
fprintf('\nPerforming ICA on CLOSED nose subjects:\n');
for i = 1:5    
    filtered_data_2d = reshape(EEG_filtered_closed(:,:,:,i), n_channels, []);
    
    %       Run ICA
    [ica_components, W_matrix, A_matrix] = ICA(filtered_data_2d, n_components, max_iterations, tolerance, learning_rate);
    ica_reshaped = reshape(ica_components, n_components, n_samples, n_trials);
    
    %       Plot ICA components PSD
    trial_for_ica = 1;
    figure;
    fprintf('    Analyzing ICA components for %s...\n', closed_names{i});
    for comp = 1:n_components
        subplot(2,2,comp);
        [psd_comp, freq_comp] = pwelch(ica_reshaped(comp, :, trial_for_ica), [], [], [], sampling_rate);
        plot(freq_comp, 10*log10(psd_comp), 'LineWidth', 1.5);
        title(sprintf('ICA Component %d - PSD', comp));
        subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
        xlabel('Frequency (Hz)'); ylabel('Power (dB)');
        xlim([0, 100]); grid on;
    end
    
    %       Plot ICA components time series
    figure;
    ica_offset = 50; hold on;
    for comp = 1:n_components
        plot(epoch_time, ica_reshaped(comp, :, trial_for_ica) + (comp-1)*ica_offset);
    end
    title(sprintf('ICA Components - Trial %d', trial_for_ica));
    subtitle(sprintf('Subject: %s (Closed Nose)', closed_names{i}));
    xlabel('Time (s)'); ylabel('Components');
    yticks((0:n_components-1)*ica_offset);
    ylim([-ica_offset, n_components*ica_offset]);
    grid on; hold off;
    
    %       Artifact removal
    artifact_components = []; % <- FILL UP
    if ~isempty(artifact_components)
        A_clean = A_matrix; 
        A_clean(:, artifact_components) = 0;
        clean_components = ica_components; 
        clean_components(artifact_components,:) = 0;
        clean_signal = A_clean * clean_components;
        clean_eeg_closed(:,:,:,i) = reshape(clean_signal, n_channels, n_samples, n_trials);
        fprintf('    Artifact components removed for %s\n', closed_names{i});
    else
        clean_eeg_closed(:,:,:,i) = EEG_filtered_closed(:,:,:,i);
        fprintf('    No artifact components removed for %s\n', closed_names{i});
    end
end


%% ICA for OPENED nose subjects
fprintf('\nPerforming ICA on OPENED nose subjects:\n');
for i = 1:5    
    filtered_data_2d = reshape(EEG_filtered_opened(:,:,:,i), n_channels, []);
    
    %      Run ICA
    [ica_components, W_matrix, A_matrix] = ICA(filtered_data_2d, n_components, max_iterations, tolerance, learning_rate);
    ica_reshaped = reshape(ica_components, n_components, n_samples, n_trials);
    
    %       Plot ICA components PSD
    trial_for_ica = 1;
    figure;
    fprintf('   ICA components for %s\n', opened_names{i});
    for comp = 1:n_components
        subplot(2,2,comp);
        [psd_comp, freq_comp] = pwelch(ica_reshaped(comp, :, trial_for_ica), [], [], [], sampling_rate);
        plot(freq_comp, 10*log10(psd_comp), 'LineWidth', 1.5);
        title(sprintf('ICA Component %d - PSD', comp));
        subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
        xlabel('Frequency (Hz)'); ylabel('Power (dB)');
        xlim([0, 100]); grid on;
    end
    
    %       Plot ICA components time series
    figure;
    ica_offset = 50; hold on;
    for comp = 1:n_components
        plot(epoch_time, ica_reshaped(comp, :, trial_for_ica) + (comp-1)*ica_offset);
    end
    title(sprintf('ICA Components - Trial %d', trial_for_ica));
    subtitle(sprintf('Subject: %s (Opened Nose)', opened_names{i}));
    xlabel('Time (s)'); ylabel('Components');
    yticks((0:n_components-1)*ica_offset);
    ylim([-ica_offset, n_components*ica_offset]);
    grid on; hold off;
    
    %       Artifact removal
    artifact_components = []; % <- FILL UP LATER
    if ~isempty(artifact_components)
        A_clean = A_matrix; 
        A_clean(:, artifact_components) = 0;
        clean_components = ica_components; 
        clean_components(artifact_components,:) = 0;
        clean_signal = A_clean * clean_components;
        clean_eeg_opened(:,:,:,i) = reshape(clean_signal, n_channels, n_samples, n_trials);
        fprintf('    Artifact components removed for %s\n', opened_names{i});
    else
        clean_eeg_opened(:,:,:,i) = EEG_filtered_opened(:,:,:,i);
        fprintf('    No artifact components removed for %s\n', opened_names{i});
    end
end

%----------------------------------------
%% 4. Feature Extraction
%----------------------------------------

fprintf('\n  4. Feature Extraction \n\n');

bands = [0.5 4; 4 8; 8 13; 13 30; 30 100];  % Delta, Theta, Alpha, Beta, Gamma
n_bands = size(bands, 1);

% Calculate feature counts
n_time_features_per_channel = 11;
n_freq_features_per_channel = 10;
n_connectivity_pairs = nchoosek(n_channels, 2);  % C(17,2) = 136
% nchoosek - binomial coefficient or all combinations

n_time_features = n_time_features_per_channel * n_channels;
n_freq_features = n_freq_features_per_channel * n_channels;
n_connectivity_features = n_connectivity_pairs * 2;

n_total_features = n_time_features + n_freq_features + n_connectivity_features; %629


all_features_closed = zeros(n_trials * 5, n_total_features);
all_features_opened = zeros(n_trials * 5, n_total_features);

% Features for CLOSED nose subjects
fprintf('\nFeatures for CLOSED nose subjects:\n');
global_trial_idx = 1;

for subj = 1:5
    fprintf('  Subject %s (closed nose)\n', closed_names{subj});
    
    for trial = 1:n_trials
        % Single trial data [channels × samples]
        if exist('clean_eeg_closed', 'var')
            trial_data = squeeze(clean_eeg_closed(:, :, trial, subj));
        else
            trial_data = squeeze(EEG_filtered_closed(:, :, trial, subj));
        end
        
        trial_features = [];
        
        % Time Domain
        time_features = [];
        for ch = 1:n_channels
            x = trial_data(ch, :);
            
            mu = mean(x);
            sigma2 = var(x);
            sigma = sqrt(sigma2);
            skew_val = skewness(x);
            kurt_val = kurtosis(x);
            
            RMS_val = sqrt(mean(x.^2));
            MAD_val = mean(abs(x - mu));
            PtP_val = max(x) - min(x);
            
            ZCR_val = sum(abs(diff(sign(x)))) / (2*(length(x)-1));
            v = diff(x);        % first difference
            a = diff(v);        % second difference
            vRMS = sqrt(mean(v.^2));
            aRMS = sqrt(mean(a.^2));
            
            time_features = [time_features, mu, sigma2, sigma, skew_val, kurt_val, ...
                           RMS_val, MAD_val, PtP_val, ZCR_val, vRMS, aRMS];
        end
        
        % Frequency Domain
        freq_features = [];
        for ch = 1:n_channels
            x = trial_data(ch, :);
            
            % PSD
            [pxx, f] = pwelch(x, [], [], [], sampling_rate);
            
            % Band powers
            band_powers = zeros(1, n_bands);
            for b = 1:n_bands
                idx = f >= bands(b,1) & f <= bands(b,2);
                band_powers(b) = sum(pxx(idx));
            end
            
            [~, peak_idx] = max(pxx);
            f_peak = f(peak_idx);
            f_mean = sum(f.*pxx)/sum(pxx);
            SC = sum(f.*pxx)/sum(pxx);  % spectral centroid
            SS = sqrt(sum(((f-SC).^2).*pxx)/sum(pxx)); % spectral spread
            P_norm = pxx/sum(pxx);
            H = -sum(P_norm.*log(P_norm + eps)); % spectral entropy
            
            freq_features = [freq_features, band_powers, f_peak, f_mean, SC, SS, H];
        end
        
        % Conectivity Features
        con_features = [];
        for i = 1:n_channels
            for j = i+1:n_channels
                % Linear correlation: channels i and j
                correlation_val = corr(trial_data(i, :)', trial_data(j, :)');
                % Coherence approximation
                coherence_val = abs(correlation_val);
                
                con_features = [con_features, correlation_val, coherence_val];
            end
        end
        
        trial_features = [time_features, freq_features, con_features];
        all_features_closed(global_trial_idx, :) = trial_features;
        global_trial_idx = global_trial_idx + 1;
    end
    
end

% Features for OPENED nose subjects
fprintf('\n Features for OPENED nose subjects:\n');
global_trial_idx = 1;

for subj = 1:5
    fprintf('  Subject %s (opened nose) \n', opened_names{subj});
    
    for trial = 1:n_trials
        if exist('clean_eeg_opened', 'var')
            trial_data = squeeze(clean_eeg_opened(:, :, trial, subj));
        else
            trial_data = squeeze(EEG_filtered_opened(:, :, trial, subj));
        end
        
        trial_features = [];
        
        % Time Domain
        time_features = [];
        for ch = 1:n_channels
            x = trial_data(ch, :);
            
            mu = mean(x);
            sigma2 = var(x);
            sigma = sqrt(sigma2);
            skew_val = skewness(x);
            kurt_val = kurtosis(x);
            
            RMS_val = sqrt(mean(x.^2));
            MAD_val = mean(abs(x - mu));
            PtP_val = max(x) - min(x);
            
            ZCR_val = sum(abs(diff(sign(x)))) / (2*(length(x)-1));
            v = diff(x);        % first difference
            a = diff(v);        % second difference
            vRMS = sqrt(mean(v.^2));
            aRMS = sqrt(mean(a.^2));
            
            time_features = [time_features, mu, sigma2, sigma, skew_val, kurt_val, ...
                           RMS_val, MAD_val, PtP_val, ZCR_val, vRMS, aRMS];
        end
        
        % Frequency Domain
        freq_features = [];
        for ch = 1:n_channels
            x = trial_data(ch, :);
            
            [pxx, f] = pwelch(x, [], [], [], sampling_rate);
            
            band_powers = zeros(1, n_bands);
            for b = 1:n_bands
                idx = f >= bands(b,1) & f <= bands(b,2);
                band_powers(b) = sum(pxx(idx));
            end
            
            [~, peak_idx] = max(pxx);
            f_peak = f(peak_idx);
            f_mean = sum(f.*pxx)/sum(pxx);
            SC = sum(f.*pxx)/sum(pxx);  % spectral centroid
            SS = sqrt(sum(((f-SC).^2).*pxx)/sum(pxx)); % spectral spread
            P_norm = pxx/sum(pxx);
            H = -sum(P_norm.*log(P_norm + eps)); % spectral entropy
            
            freq_features = [freq_features, band_powers, f_peak, f_mean, SC, SS, H];
        end
        
        % Conectivity Features
        con_features = [];
        for i = 1:n_channels
            for j = i+1:n_channels
                % Linear correlation : channels i and j
                correlation_val = corr(trial_data(i, :)', trial_data(j, :)');
                
                % Coherence approximation
                coherence_val = abs(correlation_val);
                
                con_features = [con_features, correlation_val, coherence_val];
            end
        end
        
        trial_features = [time_features, freq_features, con_features];
        all_features_opened(global_trial_idx, :) = trial_features;
        global_trial_idx = global_trial_idx + 1;
    end
    
end


all_features_combined = [all_features_closed; all_features_opened];
feature_labels = [zeros(size(all_features_closed, 1), 1); ones(size(all_features_opened, 1), 1)]; % 0=closed, 1=opened

% z-score normalization
feature_matrix_normalized = (all_features_combined - mean(all_features_combined)) ./ std(all_features_combined);

fprintf('\nFeature Extraction Summary:\n');
fprintf('   Total trials processed: %d\n', size(feature_matrix_normalized, 1));
fprintf('   Closed nose trials: %d\n', sum(feature_labels == 0));
fprintf('   Opened nose trials: %d\n', sum(feature_labels == 1));
fprintf('   Features per trial: %d\n', size(feature_matrix_normalized, 2));
fprintf('   Final feature matrix size: [%d × %d]\n', size(feature_matrix_normalized));
fprintf('   Feature breakdown:\n');
fprintf('      Time domain features: %d\n', n_time_features);
fprintf('      Frequency domain features: %d\n', n_freq_features);
fprintf('      Connectivity features: %d\n', n_connectivity_features);




%----------------------------------------------------------------------------
% Filtering Data - Analysis
% low_numbers = [0.5, 0.4, 0.3];   % High-pass cutoff frequencies
% high_numbers = [30, 35, 40];     % Low-pass cutoff frequencies
% filter_order = 2;
% 
% for i = 1:5  % loop over subjects
%     fprintf('\nSubject: %s\n', closed_names{i}); %or opened_names{i}
%     fprintf('Filter\t\tDelta\t\tAlpha\t\tBeta\t\tGamma\t\tNoise\n');
% 
%     combination_idx = 1;
%     n_filter_combinations = length(low_numbers) * length(high_numbers);
% 
%     avg_psd_results = cell(n_filter_combinations, 1);
%     filter_labels = cell(n_filter_combinations, 1);
% 
%     for z = 1:length(low_numbers)
%         for y = 1:length(high_numbers)
%             low_cutoff = low_numbers(z);
%             high_cutoff = high_numbers(y);
% 
%             filter_labels{combination_idx} = sprintf('HP=%.1f LP=%d', low_cutoff, high_cutoff);
% 
%             % Design filters
%             nyquist = sampling_rate / 2;
%             [b_hp, a_hp] = butter(filter_order, low_cutoff/nyquist, 'high');
%             [b_lp, a_lp] = butter(filter_order, high_cutoff/nyquist, 'low');
% 
%             % Apply filtering
%             EEG_filtered = zeros(size(EEG_data_closed_shaped{i}));  % [channels x (time*trials)]
%             for ch = 1:n_channels
%                 hp_output = filtfilt(b_hp, a_hp, EEG_data_closed_shaped{i}(ch, :)); %or EEG_data_opened_shaped
%                 EEG_filtered(ch, :) = filtfilt(b_lp, a_lp, hp_output);
%             end
% 
%             % Reshape to [channels x samples x trials]
%             EEG_filtered_reshaped = reshape(EEG_filtered, n_channels, n_samples, n_trials);
% 
%             % Compute PSD per trial and channel
%             all_psd = [];
%             for ch = 1:n_channels
%                 for trial = 1:n_trials
%                     trial_data = squeeze(EEG_filtered_reshaped(ch, :, trial));
%                     [psd_single, f] = pwelch(trial_data, [], [], [], sampling_rate);
%                     all_psd = [all_psd, psd_single];
%                 end
%             end
% 
%             % Compute band powers
%             delta_power = sum(all_psd(f >= 0.5 & f <= 4));
%             alpha_power = sum(all_psd(f >= 8 & f <= 13));
%             beta_power  = sum(all_psd(f >= 13 & f <= 30));
%             gamma_power = sum(all_psd(f >= 30 & f <= 45));
%             noise_level = sum(all_psd(f >= 45 & f <= 50));
% 
%             % Store results
%             avg_psd_results{combination_idx} = mean(all_psd, 2);
%             fprintf('%s\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
%                 filter_labels{combination_idx}, delta_power, alpha_power, beta_power, gamma_power, noise_level);
% 
%             combination_idx = combination_idx + 1;
%         end
%     end
% end