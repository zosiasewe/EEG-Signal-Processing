
%% Description
%{

Data set covers a project of Tasting 4 different materials
Subjects : 5 participants with Closed nose and 5 participant with opened nose
Materials used are discibed in the file "Label" as 1,2,3,4

The data from each subjects consists : 160x1280x17 meaning: trails x time x channels

The data was recorded using 512 Hz sampling frequency
Each epoch consists the recorded event of "tasting"

What we want to achieve?
- We want to perform a binary classification on both opened and closed nose
- We want to use ML algorithms for a robust classification of this closed/open nose 0/1 problem

For that we want to divide the problem into 5 subgroups:
    - Test all of the data (160 trails)
    - Test only TM1 ("Test material 1")
    - Test only TM2 ("Test material 2")
    - Test only TM3 ("Test material 3")
    - Test only TM4 ("Test material 4")


After this We want to do a research on BCI P300 and if the size of the
% screen matters 
%}

%% ---------------------------------------------------------------------------------------------------------------------------
clc
clear
close all

% Closed .mat
AC_closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
AC_closed = load(AC_closed_path);
data_AC_closed = struct2array(AC_closed);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

BA_closed_path = fullfile(pwd, 'closed_nose', 'BA.mat');
BA_closed = load(BA_closed_path);
data_BA_closed = struct2array(BA_closed);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

BO_closed_path = fullfile(pwd, 'closed_nose', 'BO.mat');
BO_closed = load(BO_closed_path);
data_BO_closed = struct2array(BO_closed);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

CMY_closed_path = fullfile(pwd, 'closed_nose', 'CMY.mat');
CMY_closed = load(CMY_closed_path);
data_CMY_closed = struct2array(CMY_closed);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

JD_closed_path = fullfile(pwd, 'closed_nose', 'JD.mat');
JD_closed = load(JD_closed_path);
data_JD_closed = struct2array(JD_closed);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

channel_names = load("Channel_names.mat");
channels = struct2array(channel_names);
closed_names = {'AC', 'BA', 'BO', 'CMY', 'JD'};

% Defininition of variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);                                           % 160 trials
n_time = size(data_AC_closed, 2);                                             % 1280 time points per trial
n_channels = size(data_AC_closed, 3);                                         % 17 channels
trial_duration_sec = n_time / sampling_rate;                        % 1280 / sampling rate
total_duration_sec = n_trials * trial_duration_sec;

data_combined = cat(4, data_AC_closed, data_BA_closed, data_BO_closed, data_CMY_closed, data_JD_closed); % [160 × 1280 × 17 × 5]

for i = 1:5
    data_perm = permute(data_combined(:,:,:,i), [3,2,1]);   % [channels × time × trials]
    EEG_data_shaped{i} = reshape(data_perm, n_channels, []); % cell array bo każdy może mieć różną długość
end


time = linspace(0, total_duration_sec, size(EEG_data_shaped{1}, 2));
offset = 300;                                                       % separation between channels

%% Plot for the whole time range

for i = 1:5 %loop for each closed nose subject 
    figure;
    hold on;
    
    for ch = 1:n_channels
        plot(time, EEG_data_shaped{i}(ch, :) + (ch - 1) * offset);
    end
    
    title('Continuous EEG plot');
    subtitle(['All 160 trials concatenated (' closed_names{i} ', closed nose)']);
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * offset);
    yticklabels(channels(1:n_channels));
    ylim([-offset, offset * n_channels]);
    xlim([0, total_duration_sec]);
    grid on;
    hold off;

%% Plot PSD for the whole set

    figure;
    
    %SNR
    snr_raw_data = zeros(1, n_channels);
    signal_band = [0.5 30];
    noise_band = [45 100];
    
    all_psds = [];
    fprintf(['\n SNR for Raw Data: ' closed_names{i} '\n']);
    for ch = 1:n_channels
        [psd_single, f] = pwelch(EEG_data_shaped{i}(ch, :), [], [], [], sampling_rate);
        all_psds = [all_psds, psd_single];
    
        signal_indices = f >= signal_band(1) & f <= signal_band(2);
        P_signal = sum(psd_single(signal_indices));
        
        noise_indices = f >= noise_band(1) & f <= noise_band(2);
        P_noise = sum(psd_single(noise_indices));
        
        snr_raw_data(ch) = 10*log10(P_signal/P_noise);
        fprintf('SNR = %.2f dB for %s\n', snr_raw_data(ch), channels{ch});
    end
    
    avg_psd = mean(all_psds, 2);
    avg_psd_db = 10*log10(avg_psd);
    
    plot(f, avg_psd_db, 'b-', 'LineWidth', 0.1,'Color',"#7E2F8E");
    title('Average PSD Across All Channels and Trials', 'FontSize', 10);
    subtitle(['(' closed_names{i} ', closed nose)']);
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    xlim([0 256]);
    ylim([-80 40]);
    grid on;



%% Plot for defined Epochs 
    epoch_num = [40, 50, 60, 70];
    time_2 = (0:n_samples-1) / sampling_rate;
    
    figure;
    
    for num_epochs = 1:length(epoch_num)
        subplot(2, 2, num_epochs);
        hold on;
    
        for ch = 1:n_channels
            plot(time_2, data_combined(epoch_num(num_epochs), :, ch,i) + (ch - 1) * offset);
        end
    
        title(['Raw EEG — Trial ', num2str(epoch_num(num_epochs))]);
        subtitle(['Before filtering (' closed_names{i} ', closed nose)']);

        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels - 1) * offset);
        yticklabels(channels(1:n_channels));
        ylim([-offset, offset * n_channels]);
        xlim([0, 2.5]);
        grid on;
        hold off;
    end
end


%% Filtering Data - Analysis
% low_numbers = [0.5, 0.4, 0.3];   % High-pass cutoff frequencies
% high_numbers = [30, 35, 40];     % Low-pass cutoff frequencies
% filter_order = 2;
% 
% for i = 1:5  % loop over subjects
%     fprintf('\nSubject: %s\n', closed_names{i});
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
%             EEG_filtered = zeros(size(EEG_data_shaped{i}));  % [channels x (time*trials)]
%             for ch = 1:n_channels
%                 hp_output = filtfilt(b_hp, a_hp, EEG_data_shaped{i}(ch, :));
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


%% Result HPF, LPF
low_cut = 0.5;
high_cut = 30;
order = 2;

EEG_filtered_result = zeros(n_channels, n_samples, n_trials, 5); 

for i = 1:5
    nyquist = sampling_rate / 2;
    [b_hp, a_hp] = butter(order, low_cut / nyquist, 'high');
    [b_lp, a_lp] = butter(order, high_cut / nyquist, 'low');
    
    EEG_filter = zeros(size(EEG_data_shaped{i}));
    
    for ch = 1:n_channels
        hp_signal = filtfilt(b_hp, a_hp, EEG_data_shaped{i}(ch, :));
        EEG_filter(ch, :) = filtfilt(b_lp, a_lp, hp_signal);
    end
    
    EEG_filtered_result(:,:,:,i) = reshape(EEG_filter, n_channels, n_samples, n_trials);
end

%% All time after filtering 
figure;
hold on;

for ch = 1:n_channels
    plot(time, EEG_filtered_result(ch, :) + (ch - 1) * offset);
end

title('Continuous EEG plot');
subtitle('After HPF, LPF (AC, closed nose)')
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
xlim([0, total_duration_sec]);
grid on;
hold off;

% --------
figure;
fprintf('\n')
fprintf('SNR for Filtered Data: \n')
%SNR
snr_raw_data_f = zeros(1, n_channels);
signal_band = [0.5 30];
noise_band = [45 100];

all_psds = [];
for ch = 1:n_channels
    [psd_single, f] = pwelch(EEG_filtered_result(ch, :), [], [], [], sampling_rate);
    all_psds = [all_psds, psd_single];

    signal_indices_f = f >= signal_band(1) & f <= signal_band(2);
    P_signal_f = sum(psd_single(signal_indices_f));
    
    noise_indices_f = f >= noise_band(1) & f <= noise_band(2);
    P_noise_f = sum(psd_single(noise_indices_f));
    
    snr_raw_data_f(ch) = 10*log10(P_signal_f/P_noise_f);
    fprintf('SNR = %.2f dB for %s\n', snr_raw_data_f(ch), channels{ch});

end

avg_psd = mean(all_psds, 2);
avg_psd_db = 10*log10(avg_psd);

plot(f, avg_psd_db, 'b-', 'LineWidth', 0.1, 'Color',"#7E2F8E");
title('Average PSD Across All Channels and Trials', 'FontSize', 10);
subtitle("After HPF and LPF Filtering (AC, closed nose)")
xlabel('Frequency (Hz)');
ylabel('PSD (dB)');
xlim([0 256]);
ylim([-80 40]);
grid on;



%% Plot for defined Epochs - after filtering
epoch_num = [40, 50, 60, 70];
time_2 = (0:n_samples-1) / sampling_rate;

figure;

for num_epochs = 1:length(epoch_num)
    subplot(2, 2, num_epochs);
    hold on;

    for ch = 1:n_channels
        plot(time_2, EEG_filtered_result(ch, :,epoch_num(num_epochs)) + (ch - 1) * offset);
    end

    title(['Raw EEG — Trial ', num2str(epoch_num(num_epochs))]);
    subtitle('After HPF, LPF (AC - closed nose)')
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels - 1) * offset);
    yticklabels(channels(1:n_channels));
    ylim([-offset, offset * n_channels]);
    xlim([0, 2.5]);
    grid on;
    hold off;
end

%% Artifact Removal - ICA

A = [];
x = EEG_filtered_result(:,:,:);
n_independent_components = 4;

x_reshaped = reshape(EEG_filtered_result, n_channels,[]);

n_independent_components = 4;  % Number of components to extract
[ica_components, W_ica, A_ica] = ICA(x_reshaped, n_independent_components, 1000, 1e-6, 1.0);

ica_reshaped = reshape(ica_components, n_independent_components, n_samples, n_trials);


%% Power Spectral Density ICA
figure();
for c = 1:n_independent_components
    subplot(2, 2, c);
    [psd_comp, f_comp] = pwelch(ica_components(c, :), [], [], [], sampling_rate);
    psd_db = 10*log10(psd_comp);
    
    plot(f_comp, psd_db, 'LineWidth', 1.5);
    title(['ICA Component ', num2str(c), ' - Power Spectral Density']);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    xlim([0, 100]);
    grid on;
end


%% Delete the artifacts
artifact_components = [];

A_clean = A_ica;
A_clean(:, artifact_components) = 0;

clean_components = ica_components;
clean_components(artifact_components, :) = 0;

clean_signal = A_clean * clean_components;
clean_eeg = reshape(clean_signal, n_channels, n_samples, n_trials);


%% Feature Extraction

