clc
clear
close all

addpath('functions');
addpath('src');
addpath('functions/functions_ES');

%------------------------------
% Params config
N_RUNS = 3;

param_configs = struct();

% 1. Different number of trees
param_configs.n_trees_list = [100, 200, 300];
param_configs.n_extracted_features_list = [25];
param_configs.k_selected_features_list = [20];
param_configs.mu_list = [20]; 
param_configs.lambda_list = [60]; 

% 2. Extracted features
% param_configs.n_trees_list = [300];
% param_configs.n_extracted_features_list = [20, 25, 30, 35];
% param_configs.k_selected_features_list = [20];
% param_configs.mu_list = [20]; 
% param_configs.lambda_list = [60]; 

% 3. Selected features
% param_configs.n_trees_list = [300];
% param_configs.n_extracted_features_list = [25];
% param_configs.k_selected_features_list = [15, 20, 25, 30];
% param_configs.mu_list = [20]; 
% param_configs.lambda_list = [60]; 

% 4. ES population sizes
% param_configs.n_trees_list = [300];
% param_configs.n_extracted_features_list = [25];
% param_configs.k_selected_features_list = [20];
% param_configs.mu_list = [10, 15, 20, 25, 30];
% param_configs.lambda_list =  3 * param_configs.mu_list;

% 5. ES offspring ratio
% param_configs.n_trees_list = [300];
% param_configs.n_extracted_features_list = [25];
% param_configs.k_selected_features_list = [20];
% param_configs.lambda_list = [40, 60, 80, 100, 120];
% param_configs.mu_list = param_configs.lambda_list / 3;

fixedParams = struct();
fixedParams.n_fuzzy_terms = 3;
fixedParams.T_max = 150;
fixedParams.k_folds_cv = 5;
fixedParams.test_ratio = 0.2;
%------------------------------

% opened
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

%closed
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

% Define Variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);
n_time = size(data_AC_closed, 2);
n_channels = size(data_AC_closed, 3);
trial_duration_sec = n_time / sampling_rate;
total_duration_sec = n_trials * trial_duration_sec;

%% Data Preprocessing 
fprintf('\n    Data Preprocessing   \n');

data_closed_combined = cat(4, data_AC_closed, data_BA_closed, data_BO_closed, data_CMY_closed, data_JD_closed);
data_opened_combined = cat(4, data_HIO_opened, data_KK_opened, data_KUA_opened, data_SK_T_opened, data_SM_opened);

% Reshape and filter data
EEG_data_closed_reshaped = cell(5,1);
EEG_data_opened_reshaped = cell(5,1);

for i = 1:5
    data_closed_permuted = permute(data_closed_combined(:,:,:,i), [3,2,1]);
    data_opened_permuted = permute(data_opened_combined(:,:,:,i), [3,2,1]);
    
    EEG_data_closed_reshaped{i} = reshape(data_closed_permuted, n_channels, []);
    EEG_data_opened_reshaped{i} = reshape(data_opened_permuted, n_channels, []);
end

% Filtering
fprintf('Filtering data.\n');
low_cutoff = 0.5;
high_cutoff = 30;
filter_order = 2;

EEG_filtered_closed = zeros(n_channels, n_samples, n_trials, 5);
EEG_filtered_opened = zeros(n_channels, n_samples, n_trials, 5);
    
for i = 1:5
    EEG_filtering_closed = EEG_data_closed_reshaped{i};
    EEG_filtered_closed(:,:,:,i) = reshape(filterEEG(EEG_filtering_closed, low_cutoff, high_cutoff, filter_order, n_channels, sampling_rate), n_channels, n_samples, n_trials);

    EEG_filtering_opened = EEG_data_opened_reshaped{i};
    EEG_filtered_opened(:,:,:,i) = reshape(filterEEG(EEG_filtering_opened, low_cutoff, high_cutoff, filter_order, n_channels, sampling_rate), n_channels, n_samples, n_trials);
end

% ICA
clean_eeg_closed = zeros(size(EEG_filtered_closed));
clean_eeg_opened = zeros(size(EEG_filtered_opened));

fprintf('Running ICA.\n');
for i = 1:5
    clean_eeg_closed(:,:,:,i) = runICASubject(EEG_filtered_closed(:,:,:,i), 4, sampling_rate, closed_names{i}, 'closed', []);
    clean_eeg_opened(:,:,:,i) = runICASubject(EEG_filtered_opened(:,:,:,i), 4, sampling_rate, opened_names{i}, 'opened', []);
end

% Raw feature extraction
bands = [0.5 4; 4 8; 8 13; 13 30; 30 100];
all_features_closed = [];
all_features_opened = [];

fprintf('Extracting raw features.\n');
for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_closed(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_closed = [all_features_closed; subj_features];
end

for subj = 1:5
    subj_features = extractRawFeatures(clean_eeg_opened(:,:,:,subj), [], n_trials, sampling_rate, bands);
    all_features_opened = [all_features_opened; subj_features];
end

all_features_combined = [all_features_closed; all_features_opened];
labels_closed = zeros(size(all_features_closed, 1), 1);
labels_opened = ones(size(all_features_opened, 1), 1);
labels = [labels_closed; labels_opened];

% Normalize features
feature_matrix_normalized = (all_features_combined - mean(all_features_combined)) ./ std(all_features_combined);

fprintf('Preprocessing completed.\n');

%% Parameter Setup Initialization

% Initialize parallel pool
if isempty(gcp('nocreate'))
    parpool('local');
end

% All parameter combinations
n_trees_list = param_configs.n_trees_list;
n_extracted_features_list = param_configs.n_extracted_features_list;
k_selected_features_list = param_configs.k_selected_features_list;
lambda_list = param_configs.lambda_list;
mu_list = param_configs.mu_list;

total_configs = length(n_trees_list) * length(n_extracted_features_list) * length(k_selected_features_list) * length(lambda_list) * length(mu_list);

fprintf('\n    Running Configurations   \n');
fprintf('Total configurations to test: %d\n', total_configs);
fprintf('Runs per configuration: %d\n', N_RUNS);

results_summary = [];
config_idx = 0;

for n_trees = n_trees_list
    for n_extracted_features = n_extracted_features_list
        for k_selected_features = k_selected_features_list
            for lambda_numbers = lambda_list
                for mu_numbers = mu_list

                    config_idx = config_idx + 1;
                    
                    fprintf('\n    Configuration %d/%d    \n', config_idx, total_configs);
                    fprintf('Trees: %d, Extracted Features: %d, Selected Features: %d, Population size: %d, Offspring size: %d\n', ...
                        n_trees, n_extracted_features, k_selected_features, mu_numbers, lambda_numbers);
                    
                    % Config
                    config_results = struct();
                    config_results.n_trees = n_trees;
                    config_results.n_extracted_features = n_extracted_features;
                    config_results.k_selected_features = k_selected_features;
                    config_results.lambda_numbers = lambda_numbers;
                    config_results.mu_numbers = mu_numbers;
                    
                    % Results
                    train_f1_runs = zeros(N_RUNS, 1);
                    test_f1_runs = zeros(N_RUNS, 1);
                    train_accuracy_runs = zeros(N_RUNS, 1);
                    test_accuracy_runs = zeros(N_RUNS, 1);
                    best_fitness_runs = zeros(N_RUNS, 1);
                    
                    for run = 1:N_RUNS
                        fprintf('  Run %d/%d: ', run, N_RUNS);
                        tic;
                        
                        % Different random seed for each run
                        random_seed = 42 + run;
                        rng(random_seed);
                        
                        % Train/Test split (stratified)
                        unique_classes = unique(labels);
                        train_idx_global = [];
                        test_idx_global = [];
                        
                        for class_label = unique_classes'
                            class_indices = find(labels == class_label);
                            n_class_samples = length(class_indices);
                            shuffled_class_idx = class_indices(randperm(n_class_samples));
                            n_test_class = round(n_class_samples * fixedParams.test_ratio);
                            
                            test_idx_global = [test_idx_global; shuffled_class_idx(1:n_test_class)];
                            train_idx_global = [train_idx_global; shuffled_class_idx(n_test_class+1:end)];
                        end
                        
                        train_idx_global = sort(train_idx_global);
                        test_idx_global = sort(test_idx_global);
                        labels_train = labels(train_idx_global);
                        labels_test = labels(test_idx_global);
                        
                        % ES Feature Extraction
                        [best_chromosome, fitness_history] = runEvolutionStrategy(...
                            feature_matrix_normalized, labels, ...
                            mu_numbers, lambda_numbers, fixedParams.T_max, ...
                            'mu_plus_lambda', n_extracted_features, size(feature_matrix_normalized, 2), ...
                            fixedParams.n_fuzzy_terms, train_idx_global);
                        
                        % Apply chromosome
                        best_features_all = applyChromosome(best_chromosome, feature_matrix_normalized, ...
                            n_extracted_features, fixedParams.n_fuzzy_terms);
                        
                        best_features_train = best_features_all(train_idx_global, :);
                        best_features_test = best_features_all(test_idx_global, :);
                        
                        % Feature Selection
                        [final_features_train, feature_indices] = featureSelection(best_features_train, labels_train, k_selected_features);
                        final_features_test = best_features_test(:, feature_indices);
                        
                        % Classification
                        final_features_combined = zeros(length(labels), size(final_features_train, 2));
                        final_features_combined(train_idx_global, :) = final_features_train;
                        final_features_combined(test_idx_global, :) = final_features_test;
                        
                        [y_pred_test, y_pred_train_cv, ~] = classifyData(final_features_combined, labels, ...
                            train_idx_global, test_idx_global, fixedParams.k_folds_cv, n_trees);
                        
                        % metrics
                        train_f1_runs(run) = calculateF1Score(labels_train, y_pred_train_cv);
                        test_f1_runs(run) = calculateF1Score(labels_test, y_pred_test);
                        train_accuracy_runs(run) = sum(labels_train == y_pred_train_cv) / length(labels_train);
                        test_accuracy_runs(run) = sum(labels_test == y_pred_test) / length(labels_test);
                        best_fitness_runs(run) = max(fitness_history);
                        test_polygon_area_runs(run) = polygonareametric(labels_test, y_pred_test);

                        
                        fprintf('Test F1=%.4f, Test Acc=%.4f (%.1fs)\n', ...
                            test_f1_runs(run), test_accuracy_runs(run), toc);
                    end
                    
                    % statistics
                    config_results.train_f1_mean = mean(train_f1_runs);
                    config_results.train_f1_std = std(train_f1_runs);
                    config_results.test_f1_mean = mean(test_f1_runs);
                    config_results.test_f1_std = std(test_f1_runs);
                    config_results.train_accuracy_mean = mean(train_accuracy_runs);
                    config_results.train_accuracy_std = std(train_accuracy_runs);
                    config_results.test_accuracy_mean = mean(test_accuracy_runs);
                    config_results.test_accuracy_std = std(test_accuracy_runs);
                    config_results.best_fitness_mean = mean(best_fitness_runs);
                    config_results.best_fitness_std = std(best_fitness_runs);
                    config_results.test_polygon_area_mean = mean(test_polygon_area_runs);
                    config_results.test_polygon_area_std = std(test_polygon_area_runs);
                    
                    % raw results
                    config_results.all_runs = struct();
                    config_results.all_runs.train_f1 = train_f1_runs;
                    config_results.all_runs.test_f1 = test_f1_runs;
                    config_results.all_runs.train_accuracy = train_accuracy_runs;
                    config_results.all_runs.test_accuracy = test_accuracy_runs;
                    config_results.all_runs.best_fitness = best_fitness_runs;
                    config_results.all_runs.test_polygon_area = test_polygon_area_runs;

                    results_summary = [results_summary; config_results];
                    
                    fprintf('  Configuration Summary:\n');
                    fprintf('    Test F1: %.4f ± %.4f\n', config_results.test_f1_mean, config_results.test_f1_std);
                    fprintf('    Test Acc: %.4f ± %.4f\n', config_results.test_accuracy_mean, config_results.test_accuracy_std);
                end
            end
        end
    end
end

%% Results

fprintf('\n\n    Results:    \n');
fprintf('%-8s %-12s %-12s %-12s %-12s %-20s %-20s %-20s %-20s %-20s\n', ...
    'Trees', 'ExtractFeat', 'SelectFeat', 'Parents', 'Offsprings', ...
    'Test F1', 'Test Accuracy', 'Train F1', 'Train Accuracy', 'Test PolygonArea');
fprintf('%s\n', repmat('-', 1, 120));


best_test_f1 = 0;
best_config_idx = 1;

for i = 1:length(results_summary)
    r = results_summary(i);
    fprintf('%-8d %-12d %-12d %-12d %-12d %-20s %-20s %-20s %-20s %-20s\n', ...
    r.n_trees, r.n_extracted_features, r.k_selected_features, ...
    r.mu_numbers, r.lambda_numbers, ...
    sprintf('%.4f±%.4f', r.test_f1_mean, r.test_f1_std), ...
    sprintf('%.4f±%.4f', r.test_accuracy_mean, r.test_accuracy_std), ...
    sprintf('%.4f±%.4f', r.train_f1_mean, r.train_f1_std), ...
    sprintf('%.4f±%.4f', r.train_accuracy_mean, r.train_accuracy_std), ...
    sprintf('%.4f±%.4f', r.test_polygon_area_mean, r.test_polygon_area_std));
    
    if r.test_f1_mean > best_test_f1
        best_test_f1 = r.test_f1_mean;
        best_config_idx = i;
    end
end

fprintf('\n   Best Configuration:   \n');
best_config = results_summary(best_config_idx);
fprintf('Trees: %d, Extracted Features: %d, Selected Features: %d, Parents size: %d, Offspring size: %d\n', ...
    best_config.n_trees, best_config.n_extracted_features, best_config.k_selected_features, ...
    best_config.mu_numbers, best_config.lambda_numbers);
fprintf('Test F1: %.4f ± %.4f\n', best_config.test_f1_mean, best_config.test_f1_std);
fprintf('Test Accuracy: %.4f ± %.4f\n', best_config.test_accuracy_mean, best_config.test_accuracy_std);
fprintf('Train F1: %.4f ± %.4f\n', best_config.train_f1_mean, best_config.train_f1_std);
fprintf('Train Accuracy: %.4f ± %.4f\n', best_config.train_accuracy_mean, best_config.train_accuracy_std);
fprintf('Test Polygon Area: %.4f ± %.4f\n', best_config.test_polygon_area_mean, best_config.test_polygon_area_std);

%Save Results
save('parameter_sweep_results.mat', 'results_summary', 'fixedParams', 'N_RUNS');
fprintf('\nResults saved \n');

figure('Position', [100, 100, 1400, 600]);

% Test F1 scores
subplot(1, 2, 1);
test_f1_means = [results_summary.test_f1_mean];
test_f1_stds = [results_summary.test_f1_std];
x_labels = cell(length(results_summary), 1);
for i = 1:length(results_summary)
    x_labels{i} = sprintf('T%d-E%d-S%d-μ%d-λ%d', results_summary(i).n_trees, ...
        results_summary(i).n_extracted_features, results_summary(i).k_selected_features, ...
        results_summary(i).mu_numbers, results_summary(i).lambda_numbers);
end

errorbar(1:length(test_f1_means), test_f1_means, test_f1_stds, 'bo-', 'LineWidth', 2);
xlabel('Configuration');
ylabel('Test F1 Score');
title('Test F1 Score Across Configurations');
set(gca, 'XTick', 1:length(test_f1_means), 'XTickLabel', x_labels);
xtickangle(45);
grid on;

%Test Accuracy
subplot(1, 2, 2);
test_acc_means = [results_summary.test_accuracy_mean];
test_acc_stds = [results_summary.test_accuracy_std];

errorbar(1:length(test_acc_means), test_acc_means, test_acc_stds, 'ro-', 'LineWidth', 2);
xlabel('Configuration');
ylabel('Test Accuracy');
title('Test Accuracy Across Configurations');
set(gca, 'XTick', 1:length(test_acc_means), 'XTickLabel', x_labels);
xtickangle(45);
grid on;

savefig('parameter_sweep_results.fig');
fprintf('Plots saved to parameter_sweep_results.fig\n');