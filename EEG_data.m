
%% Description

% Data set covers a project of Tasting 4 different materials
% Subjects : 5 participants with Closed nose and 5 participant with opened nose
% Materials used are discibed in the file "Label" as 1,2,3,4

% The data from each subjects consists : 160x1280x17 meaning: trails x time x channels

% What we want to achieve?
% We want to perform a binary classification o both opened and closed eyes
% We want to use ML algorithms for a robust classification of this
% closed/open eyes 0/1 problem
% For that we want to divide the problem into 5 subgroups:
    % Test all of the data (160 trails)
    % Test only TM1 ("Test material 1")
    % Test only TM2 ("Test material 2")
    % Test only TM3 ("Test material 3")
    % Test only TM4 ("Test material 4")


% After this We want to do a research on BCI P300 and if the size of the
% screen matters 


%%Trail 2 

clc
clear
close all

closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
tmp = load(closed_path);
data = struct2array(tmp);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]
channel_names = load("Channel_names.mat");
channels = struct2array(channel_names);

% Defininition of variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data, 1);                                           % 160 trials
n_time = size(data, 2);                                             % 1280 time points per trial
n_channels = size(data, 3);                                         % 17 channels
trial_duration_sec = n_time / sampling_rate;                        % 1280 / sampling rate
total_duration_sec = n_trials * trial_duration_sec;


data_perm = permute(data, [3, 2, 1]);                               % [channels × time × trials]
EEG_data_shaped = reshape(data_perm, n_channels, []);               % [17 × (1280*160)]

time = linspace(0, total_duration_sec, size(EEG_data_shaped, 2));   % time for entire signa
offset = 300;                                                       % separation between channels

%% Plot for the whole time range
figure(1);
hold on;

for ch = 1:n_channels
    plot(time, EEG_data_shaped(ch, :) + (ch - 1) * offset);
end

title('Continuous EEG plot — all 160 trials concatenated (AC, closed nose)');
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
xlim([0, total_duration_sec]);
grid on;
hold off;


%% Plot for defined Epochs 
epoch_num = 1;
EEG_epoched = squeeze(data(epoch_num,:,:)); % 1280 x 17

time_2 = (0:n_samples-1) / sampling_rate;

figure(2);
hold on;

for ch = 1:n_channels
    plot(time_2, (EEG_epoched(:, ch) + (ch - 1) * offset));
end

title('Partly EEG plot — Per epochs plot (AC, closed nose)');
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
% xlim([0, total_duration_sec/sampling_rate]);
grid on;
hold off;


%% Filtering the data

% Low pass filter 

% Use Butterworth Filters butter
% Use filtfilt() for zero-phase filtering (no time delay)
% Filter each channel separately to avoid cross-channel artifacts
% Apply high-pass first, then low-pass
% High-pass: 0.5 Hz instead of 0.1 Hz (removes DC drift without instability)
% Low-pass: 30 Hz (same as yours, good for EEG)

nyquist = sampling_rate / 2;
low_cutoff = 0.5;
high_cutoff = 30;
filter_order = 2;

[b_hp, a_hp] = butter(filter_order, low_cutoff / nyquist, 'high');
[b_lp, a_lp] = butter(filter_order, high_cutoff / nyquist, 'low');

EEG_filtered = zeros(size(EEG_data_shaped));

for ch = 1:n_channels
    hp_output = filtfilt(b_hp, a_hp, EEG_data_shaped(ch, :));
    EEG_filtered(ch, :) = filtfilt(b_lp, a_lp, hp_output);
end

filt_perm = permute(EEG_filtered, [2, 1]);


EEG_filtered_reshaped = reshape(EEG_filtered, n_channels, n_samples, n_trials);  % [17 × 1280 × 160]

trial_num = 1;
EEG_filtered_epoched = squeeze(EEG_filtered_reshaped(:, :, trial_num));

time_2 = (0:n_samples-1) / sampling_rate;

figure(3);
hold on;

for ch = 1:n_channels
    plot(time_2, (EEG_filtered_epoched(ch, :) + (ch - 1) * offset));
end

title('Partly EEG plot — Per epochs plot (AC, closed nose)');
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
% xlim([0, total_duration_sec/sampling_rate]);
grid on;
hold off;


% figure(3);
% hold on;
% 
% for ch = 1:n_channels
%     plot(time, hpf(ch, :) + (ch - 1) * offset);
% end
% 
% title('Continuous EEG plot — all 160 trials concatenated (AC, closed nose)');
% xlabel('Time (s)');
% ylabel('Channels');
% yticks((0:n_channels-1) * offset);
% yticklabels(channels(1:n_channels));
% ylim([-offset, offset * n_channels]);
% xlim([0, total_duration_sec]);
% grid on;
% hold off;


% High pass filter 






