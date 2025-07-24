clc
clear all 
close all

% Data preparation

closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
% lalala
tmp = load(closed_path);
data = struct2array(tmp);  % teraz: [160 × 1280 × 17]

data_perm = permute(data, [3, 2, 1]);  % → [channels × time × trials] = [17 × 1280 × 160]
EEG_data = reshape(data_perm, 17, []);  % → [17 × (1280 * 160)] = [17 × 204800]
% 1280 punktów czasowych/trial × 160 trials = 204800 próbek na kanał

eeglab;

%% Continuos data
EEG_continuous = pop_importdata('dataformat','array','nbchan', 17, 'data', EEG_data,'srate', 256,'pnts', size(EEG_data,2),'xmin', 0);
EEG_continuous = eeg_checkset(EEG_continuous);
[ALLEEG, EEG_continuous, CURRENTSET] = eeg_store(ALLEEG, EEG_continuous, 1);

pop_eegplot(EEG_continuous, 1, 1, 1);

%% Data per epochs
EEG_epoched = eeg_emptyset();
EEG_epoched.data = data_perm; % [17 × 1280 × 160]
EEG_epoched.nbchan = 17;
EEG_epoched.pnts = 1280; % 5 seconds per trial
EEG_epoched.trials = 160;% 160  trials
EEG_epoched.srate = 256;
EEG_epoched.xmin = 0;
EEG_epoched.xmax = (1280-1)/256;
EEG_epoched.times = (0:1279) * 1000/256; % time vector in ms
EEG_epoched = eeg_checkset(EEG_epoched);

[ALLEEG, EEG_epoched, CURRENTSET] = eeg_store(ALLEEG, EEG_epoched, 2);

pop_eegplot(EEG_epoched, 1, 1, 1);




