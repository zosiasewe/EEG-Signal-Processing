
%% Description

% Data set covers a project of Tasting 4 different materials
% Subjects : 5 participants with Closed nose and 5 participant with opened nose
% Materials used are discibed in the file "Label" as 1,2,3,4

% The data from each subjects consists : 160x1280x17 meaning: trails x time x channels

% The data was recorded using 512 Hz sampling frequency
% Each epoch consists the recorded event of "tasting"

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

%% ---------------------------------------------------------------------------------------------------------------------------
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
epoch_num = [20, 21, 22, 23];
time_2 = (0:n_samples-1) / sampling_rate;

figure(2);

for num_epochs = 1:length(epoch_num)
    subplot(2, 2, num_epochs);
    hold on;

    for ch = 1:n_channels
        plot(time_2, data(epoch_num(num_epochs), :, ch) + (ch - 1) * offset);
    end

    title(['Raw EEG — Trial ', num2str(epoch_num(num_epochs))]);
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels - 1) * offset);
    yticklabels(channels(1:n_channels));
    ylim([-offset, offset * n_channels]);
    xlim([0, 2.5]);
    grid on;
    hold off;
end



%% Filtering Data - Fixed Version
low_numbers = [0.5, 0.4, 0.3];
high_numbers = [30, 35, 40];

combination_idx = 1;
n_filter_combinations = length(low_numbers) * length(high_numbers);
filter_labels = cell(n_filter_combinations, 1);
avg_psd_results = cell(n_filter_combinations, 1);
colors = lines(n_filter_combinations);

%Stats
results = [];
filter_names = {};
counter = 1;

figure;
hold on;

for z = 1:length(low_numbers)
    for y = 1:length(high_numbers)
        
        nyquist = sampling_rate / 2;
        low_cutoff = low_numbers(z);
        high_cutoff = high_numbers(y);
        filter_order = 2;
        
        filter_labels{combination_idx} = sprintf('HP=%.1f LP=%d', low_cutoff, high_cutoff);
        
        [b_hp, a_hp] = butter(filter_order, low_cutoff / nyquist, 'high');
        [b_lp, a_lp] = butter(filter_order, high_cutoff / nyquist, 'low');
        
        EEG_filtered = zeros(size(EEG_data_shaped));
        
        for ch = 1:n_channels
            hp_output = filtfilt(b_hp, a_hp, EEG_data_shaped(ch, :));
            EEG_filtered(ch, :) = filtfilt(b_lp, a_lp, hp_output);
        end
        
        EEG_filtered_reshaped = reshape(EEG_filtered, n_channels, n_samples, n_trials);
        
        all_psd = [];
        
        for chan = 1:n_channels
            for trial = 1:n_trials
                
                trial_data = squeeze(EEG_filtered_reshaped(chan, :, trial));  % Single trail
                
                [psd_single, f] = pwelch(trial_data, [], [], [], sampling_rate);
                
                if isempty(all_psd)
                    all_psd = psd_single;
                    freq_vector = f;
                else
                    all_psd = [all_psd, psd_single];
                end
            end
        end
        

        delta_power = sum(all_psd(f >= 0.5 & f <= 4));
        alpha_power = sum(all_psd(f >= 8 & f <= 13));
        beta_power = sum(all_psd(f >= 13 & f <= 30));
        gamma_power = sum(all_psd(f >= 30 & f <= 45));
        noise_level = sum(all_psd(f >= 45 & f <= 50));

        avg_psd = mean(all_psd, 2);
        avg_psd_results{combination_idx} = avg_psd;
        
        semilogy(freq_vector, avg_psd, 'Color', colors(combination_idx, :), 'LineWidth', 2);
        
        combination_idx = combination_idx + 1;

        results(counter, :) = [delta_power, alpha_power, beta_power, gamma_power, noise_level];
        filter_names{counter} = sprintf('HP=%.1f LP=%d', low_cutoff, high_cutoff);
        
        fprintf('%s\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n', ...
            filter_names{counter}, delta_power, alpha_power, beta_power, gamma_power, noise_level);
        
        counter = counter + 1;

    end
end

title('Average PSD Across All Channels and Trials');
xlabel('Frequency (Hz)');
ylabel('PSD (μV²/Hz)');
xlim([0 50]);
legend(filter_labels, 'Location', 'best');
grid on;
hold off;


%% We choose 