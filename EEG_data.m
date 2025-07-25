
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



%% Filtering the data

% Testing the data for different Filter Values
low_numbers = [0.5, 0.4, 0.3];
high_numbers = [30, 35, 40];

for z=1:length(low_numbers)
    for y=1:length(high_numbers)
        
        nyquist = sampling_rate / 2;
        low_cutoff = low_numbers(1,z);
        high_cutoff = high_numbers(1,y);
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
        
        trial_nums = [20, 21, 22, 23];
        
        figure_num = 100 + (z - 1) * length(high_numbers) + y;
        figure(figure_num);        

        for num = 1:length(trial_nums)
            subplot(2, 2, num);
            
            EEG_filtered_epoched_1 = squeeze(EEG_filtered_reshaped(:, :, trial_nums(1)));
            EEG_filtered_epoched_2 = squeeze(EEG_filtered_reshaped(:, :, trial_nums(2)));
            EEG_filtered_epoched_3 = squeeze(EEG_filtered_reshaped(:, :, trial_nums(3)));
            EEG_filtered_epoched_4 = squeeze(EEG_filtered_reshaped(:, :, trial_nums(4)));           

            time_2 = (0:n_samples - 1) / sampling_rate;
            hold on;
        
            for ch = 1:n_channels
                plot(time_2, (EEG_filtered_epoched_1(ch, :) + (ch - 1) * offset), time_2, (EEG_filtered_epoched_1(ch, :) + (ch - 1) * offset), '--', ...
                    time_2, (EEG_filtered_epoched_1(ch, :) + (ch - 1) * offset), '--', time_2, (EEG_filtered_epoched_1(ch, :) + (ch - 1) * offset), '--');
            end
        
            title(['Filtered EEG — Trial ', num2str(trial_nums(num))]);
            subtitle(['Filter: HP = ', num2str(low_cutoff), ' Hz, LP = ', num2str(high_cutoff), ' Hz']);
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
end



%% Filtering Data - different approach


% we want to maybe ((((??????))))) plot it on a one figure (4 subplots but
% different colors line etc to see the differences.

% Testing the data for different Filter Values
low_numbers = [0.5, 0.4, 0.3];
high_numbers = [30, 35, 40];

for z=1:length(low_numbers)
    for y=1:length(high_numbers)
        
        nyquist = sampling_rate / 2;
        low_cutoff = low_numbers(1,z);
        high_cutoff = high_numbers(1,y);
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
        
        trial_nums = [20, 21, 22, 23];
        
        figure(100);        

        for num = 1:length(trial_nums)
            subplot(2, 2, num);
            
            EEG_filtered_epoched = squeeze(EEG_filtered_reshaped(:, :, trial_nums(num)));
            
            time_2 = (0:n_samples - 1) / sampling_rate;
            hold on;
        
            for ch = 1:n_channels
                plot(time_2, EEG_filtered_epoched(ch, :) + (ch - 1) * offset);
            end
        
            title(['Filtered EEG — Trial ', num2str(trial_nums(num))]);
            subtitle(['Filter: HP = ', num2str(low_cutoff), ' Hz, LP = ', num2str(high_cutoff), ' Hz']);
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
end




