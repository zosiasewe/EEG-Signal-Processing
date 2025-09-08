%% Description
%{
Data set covers a project of Tasting Process
Subjects : 5 participants with Closed nose and 5 participant with opened nose

The data from each subjects consists : 160x1280x17 meaning: trials x time x channels

The data was recorded using 512 Hz sampling frequency
Each epoch consists the recorded event of "tasting"

- We want to perform a binary classification on both opened and closed nose

%}

%% ---------------------------------------------------------------------------------------------------------------------------
clc
clear
close all

addpath('functions');
addpath('src');
addpath('functions/functions_ES');

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

% ----------
% Define Variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);                    % 160 trials
n_time = size(data_AC_closed, 2);                      % 1280 time points per trial
n_channels = size(data_AC_closed, 3);                  % 17 channels
trial_duration_sec = n_time / sampling_rate;           % Trial duration in seconds
total_duration_sec = n_trials * trial_duration_sec;    % Total recording duration
% ----------

% Combine and Reshape Data

data_closed_combined = cat(4, data_AC_closed, data_BA_closed, data_BO_closed, data_CMY_closed, data_JD_closed);
data_opened_combined = cat(4, data_HIO_opened, data_KK_opened, data_KUA_opened, data_SK_T_opened, data_SM_opened);

% Reshape data for plotting - continuous
EEG_data_closed_reshaped = cell(5,1);
EEG_data_opened_reshaped = cell(5,1);

for i = 1:5
    data_closed_permuted = permute(data_closed_combined(:,:,:,i), [3,2,1]);   % [channels × time × trials]
    data_opened_permuted = permute(data_opened_combined(:,:,:,i), [3,2,1]);
    
    EEG_data_closed_reshaped{i} = reshape(data_closed_permuted, n_channels, []); % Continuous
    EEG_data_opened_reshaped{i} = reshape(data_opened_permuted, n_channels, []);
end

time_axis = linspace(0, total_duration_sec, size(EEG_data_closed_reshaped{1}, 2));
channel_offset = 300;


%----------------------------------------
%% 1. Raw Data 
%----------------------------------------
% Raw Data Plots & PSD plots & SNR calculations & Epoch Plots

fprintf('\n   1. Raw Data \n\n');
fprintf('\n Closed nose subjects:\n');
for i = 1:5
    EEG_data_closed = EEG_data_closed_reshaped{i};
%     plotContinuousEEG(time_axis, EEG_data_closed, n_channels, channel_offset, closed_names{i}, 'closed', 'raw', channels);
    
    PSD_data_subj = EEG_data_closed_reshaped{i};
    analyzePSD(PSD_data_subj,sampling_rate, n_channels, channels, closed_names{i}, 'closed', 'raw')

end

fprintf('\n Opened nose subjects:\n');
for i = 1:5
    EEG_data_subj_opened = EEG_data_opened_reshaped{i};
%     plotContinuousEEG(time_axis, EEG_data_subj_opened, n_channels, channel_offset, opened_names{i}, 'opened', 'raw', channels);
    
    PSD_data_subj_opened = EEG_data_opened_reshaped{i};
    analyzePSD(PSD_data_subj_opened,sampling_rate, n_channels, channels, opened_names{i}, 'opened', 'raw')

end

epoch_samples = [40, 50, 60, 70];
epoch_time = (0:n_samples-1) / sampling_rate;

for i = 1:5
    EEG_data_closed = data_closed_combined(:, :, :, i); % all trials for this subject
%     plotIndividualEpochs(EEG_data_closed, epoch_samples, epoch_time, closed_names{i}, 'closed', n_channels, channel_offset, channels, 'raw');
    
    EEG_data_opened = data_closed_combined(:, :, :, i); % all trials for this subject
%     plotIndividualEpochs(EEG_data_opened, epoch_samples, epoch_time, opened_names{i}, 'opened', n_channels, channel_offset, channels, 'raw');
end


%----------------------------------------
%% 2. Filtering
%----------------------------------------

fprintf('\n   2. Filtering \n\n');

% Filter params
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


% Continuous plot & PSD plot & SNR & Epochs
for i = 1:5
    Closed_filtered_continuous = reshape(EEG_filtered_closed(:,:,:,i), n_channels, []);
%     plotContinuousEEG(time_axis, Closed_filtered_continuous, n_channels, channel_offset, closed_names{i}, 'closed', 'filt', channels);
    analyzePSD(Closed_filtered_continuous, sampling_rate, n_channels, channels, closed_names{i}, 'closed', 'filt');
end

for i = 1:5
    Opened_filtered_continuous = reshape(EEG_filtered_opened(:,:,:,i), n_channels, []);
%     plotContinuousEEG(time_axis, Opened_filtered_continuous, n_channels, channel_offset, opened_names{i}, 'opened', 'filt', channels);
    analyzePSD(Opened_filtered_continuous, sampling_rate, n_channels, channels, opened_names{i}, 'opened', 'filt');
end

epoch_samples = [40, 50, 60, 70];
for i = 1:5
    Epoch_closed_data = zeros(n_channels, n_samples, length(epoch_samples));
    for t_idx = 1:length(epoch_samples)
        trial_num = epoch_samples(t_idx);
        Epoch_closed_data(:, :, t_idx) = EEG_filtered_closed(:, :, trial_num, i);
    end
%     plotIndividualEpochs(Epoch_closed_data, epoch_samples, epoch_time, closed_names{i}, 'closed', n_channels, channel_offset, channels, 'filt');
end

for i = 1:5
    Epoch_opened_data = zeros(n_channels, n_samples, length(epoch_samples));
    for t_idx = 1:length(epoch_samples)
        trial_num = epoch_samples(t_idx);
        Epoch_opened_data(:, :, t_idx) = EEG_filtered_opened(:, :, trial_num, i);
    end
%     plotIndividualEpochs(Epoch_opened_data, epoch_samples, epoch_time, opened_names{i}, 'opened', n_channels, channel_offset, channels, 'filt');
end

%----------------------------------------
%% 3. Artifact removal - ICA
%----------------------------------------

fprintf('\n   3. ICA \n\n');
n_components = 4;
max_iterations = 1000;
tolerance = 1e-6;
learning_rate = 1.0;

clean_eeg_closed = zeros(size(EEG_filtered_closed));
for i = 1:5
    fprintf('ICA on Closed Nose Subject: %s\n', closed_names{i});
    clean_eeg_closed(:,:,:,i) = runICASubject(EEG_filtered_closed(:,:,:,i), 4, sampling_rate, closed_names{i}, 'closed', epoch_time);
end

clean_eeg_opened = zeros(size(EEG_filtered_opened));
for i = 1:5
    fprintf('ICA on Opened Nose Subject: %s\n', opened_names{i});
    clean_eeg_opened(:,:,:,i) = runICASubject(EEG_filtered_opened(:,:,:,i), 4, sampling_rate, opened_names{i}, 'opened', epoch_time);
end

%----------------------------------------
%% 4. Raw Feature Extraction
%----------------------------------------
fprintf('\n   4. Raw Feature Extraction \n\n');
bands = [0.5 4; 4 8; 8 13; 13 30; 30 100]; % freq bands
n_trials = size(data_AC_closed, 1); % 160 trials
sampling_rate = 512;

all_features_closed = [];
all_features_opened = [];

fprintf('\nFeatures for Closed nose subjects.\n');
for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_closed(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_closed = [all_features_closed; subj_features];
end

fprintf('\nFeatures for Opened nose subjects.\n');
for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_opened(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_opened = [all_features_opened; subj_features];
end

all_features_combined = [all_features_closed; all_features_opened];


labels_closed = zeros(size(all_features_closed, 1), 1);  % closed nose = 0
labels_opened = ones(size(all_features_opened, 1), 1);  % opened nose = 1
labels = [labels_closed; labels_opened];

% z-score normalization
feature_matrix_normalized = (all_features_combined - mean(all_features_combined)) ./ std(all_features_combined);

%----------------------------------------
%% 5. Splitting Data
%----------------------------------------
fprintf('\n   5. Splitting Data \n\n');

random_seed = 42;
rng(random_seed);

test_ratio = 0.2; 
stratified = true; % class balance

fprintf('Test ratio: %.1f%%\n', test_ratio * 100);

% stratified split
unique_classes = unique(labels);
train_idx_global = [];
test_idx_global = [];

for class_label = unique_classes'
    class_indices = find(labels == class_label);
    n_class_samples = length(class_indices);
    
    fprintf('Class %d: %d samples\n', class_label, n_class_samples);
    
    % Shuffle class indices
    shuffled_class_idx = class_indices(randperm(n_class_samples));
    
    % Split this class
    n_test_class = round(n_class_samples * test_ratio);
    n_train_class = n_class_samples - n_test_class;
    
    test_idx_global = [test_idx_global; shuffled_class_idx(1:n_test_class)];
    train_idx_global = [train_idx_global; shuffled_class_idx(n_test_class+1:end)];
    
    fprintf('     Train: %d, Test: %d\n', n_train_class, n_test_class);
end

train_idx_global = sort(train_idx_global);
test_idx_global = sort(test_idx_global);

labels_train = labels(train_idx_global);
labels_test = labels(test_idx_global);
fprintf('Training class distribution:\n');
for class_label = unique_classes'
    n_class_train = sum(labels_train == class_label);
    fprintf('  Class %d: %d (%.1f%%)\n', class_label, n_class_train, 100*n_class_train/length(labels_train));
end

fprintf('Testing class distribution:\n');
for class_label = unique_classes'
    n_class_test = sum(labels_test == class_label);
    fprintf('  Class %d: %d (%.1f%%)\n', class_label, n_class_test, 100*n_class_test/length(labels_test));
end

% Verify overlap
if ~isempty(intersect(train_idx_global, test_idx_global))
    error('!Train and test indices overlap!');
else
    fprintf(' No overlap between train and test sets\n');
end

%----------------------------------------
%% 6. ES - Fuzzy Feature Extraction
%----------------------------------------
fprintf('\n   6. ES - Fuzzy Feature Extraction \n\n');

if isempty(gcp('nocreate'))
    parpool('local');
end

% ES Parameters
n_extracted_features = 25;
n_fuzzy_terms = 3;
feature_pool_size = size(feature_matrix_normalized, 2);

mu = 20; % population size
lambda = 3*mu; % offspring size
T_max = 150; % max generations
selection_mode = 'mu_plus_lambda';

fprintf('  Population: %d, Offspring: %d, Max Gen: %d\n', mu, lambda, T_max);
fprintf('  Features to extract: %d from %d raw features\n', n_extracted_features, feature_pool_size);
fprintf('\n \n');

[best_chromosome, fitness_history] = runEvolutionStrategy(feature_matrix_normalized, labels, ...
    mu, lambda, T_max, selection_mode, n_extracted_features, feature_pool_size, n_fuzzy_terms, train_idx_global);

fprintf('Best fitness achieved: %.4f\n', max(fitness_history));

% Best chromosome to train and test data
best_features_all = applyChromosome(best_chromosome, feature_matrix_normalized, n_extracted_features, n_fuzzy_terms);

% train and test info
best_features_train = best_features_all(train_idx_global, :);
best_features_test = best_features_all(test_idx_global, :);

fprintf('\nGenerated ES fuzzy features:\n');
fprintf('  Training features: %d x %d\n', size(best_features_train));
fprintf('  Testing features: %d x %d\n', size(best_features_test));
fprintf('  Total features generated: %d\n', size(best_features_all, 2));

%----------------------------------------
%% 8. Feature Selection
%----------------------------------------
fprintf('\n   8. Feature Selection \n\n');

k = 20; 
fprintf('Selecting %d best features using training data.\n', k);

% Feature selection based on training data
[final_features_train, feature_indices] = featureSelection(best_features_train, labels_train, k);

% Apply same feature selection to test data
final_features_test = best_features_test(:, feature_indices);


%----------------------------------------
%% 9. Classification Train/Test Split
%----------------------------------------
fprintf('\n   9. Classification \n\n');

% full feature matrix with selected features 
final_features_combined = zeros(length(labels), size(final_features_train, 2));
final_features_combined(train_idx_global, :) = final_features_train;
final_features_combined(test_idx_global, :) = final_features_test;

k_folds_cv = 5; 
n_trees = 300;

[y_pred_test, y_pred_train_cv, final_rf_model] = ...
    classifyData(final_features_combined, labels, train_idx_global, test_idx_global, k_folds_cv, n_trees);

%----------------------------------------
%% 10. Results
%----------------------------------------
fprintf('\n   10. Results \n\n');

% Training performance
train_f1 = calculateF1Score(labels_train, y_pred_train_cv);
train_accuracy = sum(labels_train == y_pred_train_cv) / length(labels_train);

% Test performance
test_f1 = calculateF1Score(labels_test, y_pred_test);
test_accuracy = sum(labels_test == y_pred_test) / length(labels_test);

% Polygon area metric
test_metric = polygonareametric(labels_test, y_pred_test);

figure('Position', [100, 100, 1200, 400]);
subplot(1, 2, 1);
confusionchart(labels_train, y_pred_train_cv, 'Title', 'Training Set (CV)');
subplot(1, 2, 2);
confusionchart(labels_test, y_pred_test, 'Title', 'Test Set (Unseen)');


fprintf('\n\n');

fprintf('Dataset: EEG Taste Classification\n');
fprintf('Task: Closed/Open Nose Classification\n');
fprintf('Random Seed: %d\n', random_seed);
fprintf('\n Dataset Split \n');
fprintf('Total Samples: %d\n', length(labels));
fprintf('Training: %d (%.1f%%)\n', length(train_idx_global), 100*length(train_idx_global)/length(labels));
fprintf('Testing: %d (%.1f%%)\n', length(test_idx_global), 100*length(test_idx_global)/length(labels));

fprintf('\n Feature Engineering \n');
fprintf('Raw features: %d\n', size(feature_matrix_normalized, 2));
fprintf('ES-generated features: %d\n', size(best_features_all, 2));
fprintf('Selected features: %d\n', size(final_features_train, 2));
fprintf('ES generations: %d\n', length(fitness_history));

fprintf('\n Methodology \n');
fprintf('ES optimization: Training data only (%d samples)\n', length(train_idx_global));
fprintf('Feature selection: Training data only\n');
fprintf('Cross-validation: %d-fold on training data\n', k_folds_cv);
fprintf('Final evaluation: Held-out test set\n');

fprintf('\n Performance Results \n');
fprintf('Training F1 (CV): %.4f\n', train_f1);
fprintf('Training Accuracy (CV): %.4f\n', train_accuracy);
fprintf('Test F1: %.4f\n', test_f1);
fprintf('Test Accuracy: %.4f\n', test_accuracy);

fprintf('\n Model Details \n');
fprintf('Random Forest Trees: %d\n', n_trees);
fprintf('ES Population: %d\n', mu);
fprintf('ES Offspring: %d\n', lambda);
fprintf('Final Best Fitness: %.4f\n', max(fitness_history));

%Saving
results = struct();
results.random_seed = random_seed;
results.best_chromosome = best_chromosome;
results.fitness_history = fitness_history;
results.tree_number = n_trees;
results.train_f1 = train_f1;
results.test_f1 = test_f1;
results.train_accuracy = train_accuracy;
results.test_accuracy = test_accuracy;
results.polygon_metric = test_metric;
results.n_raw_features = size(feature_matrix_normalized, 2);
results.n_es_features = size(best_features_all, 2);
results.n_selected_features = size(final_features_train, 2);
results.methodology = 'ES Fuzzy Output';

save('publication_results.mat', 'results');

fprintf('Results saved \n');
