
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


%% Data loading 
% clc
% clear all
% close all
% % 
% 
% closed_path = fullfile(pwd, 'closed_nose');
% opened_path = fullfile(pwd, 'opened_nose'); %pwd - current folder
% load_single_var = @(fpath) struct2array(load(fpath));
% 
% % 
% % % Closed nose data
% AC_closed = load_single_var(fullfile(closed_path, "AC.mat"));
% BA_closed = load_single_var(fullfile(closed_path, "BA.mat"));
% BO_closed = load_single_var(fullfile(closed_path, "BO.mat"));
% CMY_closed = load_single_var(fullfile(closed_path, "CMY.mat"));
% JD_closed = load_single_var(fullfile(closed_path, "JD.mat"));
% Labels_closed = load_single_var(fullfile(closed_path, "Label_closed.mat"));
% % 
% % Opened nose data
% HIO_opened    = load_single_var(fullfile(opened_path, "HIO.mat"));
% KK_opened     = load_single_var(fullfile(opened_path, "KK.mat"));
% KUA_opened    = load_single_var(fullfile(opened_path, "KUA.mat"));
% SKT_opened    = load_single_var(fullfile(opened_path, "SK_T.mat"));
% SM_opened     = load_single_var(fullfile(opened_path, "SM.mat"));
% Labels_opened = load_single_var(fullfile(opened_path, "Label_opened.mat"));
% 
% 
%  
% % AC eyes closed : 1 trail 
% AC_C_1 = squeeze(AC_closed(1,:,:)); % first trail
% frequency = 250; %sampling frequency
% time = (0:size(AC_C_1,1)-1) / frequency;
% offset = 200;
% 
% figure(1)
% hold on;
% for channels = 1:size(AC_C_1, 2)
%     plot(time, AC_C_1(:,channels) + (channels-1)*offset);
% end
% 
% title("AC - eyes closed I trail")
% ylabel("channels")
% xlabel("time (s)");
% xlim([0 5.12])
% ylim([-100 3370])
% yticks((0:size(AC_C_1,2)-1) * offset);
% yticklabels({"CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8","CH9","CH10","CH11","CH12","CH13","CH14","CH15","CH16","CH17"});
% grid on;
% hold off;


% closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
% % lalala
% tmp = load(closed_path);
% data = struct2array(tmp);  % teraz: [160 × 1280 × 17]
% offset = 256;
% data_perm = permute(data, [3, 2, 1]);  % → [channels × time × trials] = [17 × 1280 × 160]
% EEG_data_shaped = reshape(data_perm, 17, []);  % → [17 × (1280 * 160)] = [17 × 204800]
% 
% figure(1)
% hold on;
% for channels = 1:size(EEG_data_shaped, 2)
%     plot(time, EEG_data_shaped(:,channels) + (channels-1)*offset);
% end
% 
% grid on;
% hold off;

% all trails one channel


%%Trail 2 

clc
clear
close all

closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
tmp = load(closed_path);
data = struct2array(tmp);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]

sampling_rate = 256;
n_samples = 1280;
n_trials = size(data, 1);                                           % 160 trials
n_time = size(data, 2);                                             % 1280 time points per trial
n_channels = size(data, 3);                                         % 17 channels
trial_duration_sec = n_time / sampling_rate;                        % 1280 / sampling rate
total_duration_sec = n_trials * trial_duration_sec;

Channel_names = ["F4" "F3" "F8" "F7" "FZ" "C4" "C3" "T4" "T3" "T6" "T5" "P4" "P3" "PZ" "O2" "O1" "OZ"];


data_perm = permute(data, [3, 2, 1]);                               % [channels × time × trials]
EEG_data_shaped = reshape(data_perm, n_channels, []);               % [17 × (1280*160)]

time = linspace(0, total_duration_sec, size(EEG_data_shaped, 2));   % time for entire signal

offset = 300;                                                       % vertical separation between channels


figure(1);
hold on;

for ch = 1:n_channels
    plot(time, EEG_data_shaped(ch, :) + (ch - 1) * offset);
end

title('Continuous EEG plot — all 160 trials concatenated (AC, closed nose)');
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
ylim([-offset, offset * n_channels]);
xlim([0, total_duration_sec]);
grid on;
hold off;


% Plot for epochs 

EEG_epoched = squeeze(data(1,:,:)); % 1280 x 17

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
ylim([-offset, offset * n_channels]);
% xlim([0, total_duration_sec/sampling_rate]);
grid on;
hold off;







