clc
clear
close all

% ========== CONFIGURATION SECTION ==========
% Set which parameter sets to run
RUN_SETS = {
    % 'TreesComparison',        % Comment out if already done
    % 'ExtractedFeatures',      % Comment out if already done  
    % 'SelectedFeatures',       % Comment out if already done
    'PopulationSizes',          % Uncomment to run this set
    % 'OffspringRatio',         % Uncomment to run this set
    % 'TestRatioComparison'     % Uncomment to run this set
};

% ========== PARTIAL RUN SETTINGS ==========
TOTAL_RUNS = 20;              % Total runs you want for each config
RUNS_PER_BATCH = 5;           % How many runs to do in this execution
CURRENT_BATCH = 1;            % Which batch is this? (1, 2, 3, or 4)

% Calculate which runs to execute this time
start_run = (CURRENT_BATCH - 1) * RUNS_PER_BATCH + 1;
end_run = min(CURRENT_BATCH * RUNS_PER_BATCH, TOTAL_RUNS);
runs_this_batch = start_run:end_run;

fprintf('BATCH %d: Running %d runs (%d to %d)\n', CURRENT_BATCH, length(runs_this_batch), start_run, end_run);
% ===========================================

timestamp_str = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
batch_str = sprintf('batch%d', CURRENT_BATCH);

fprintf('Starting parameter tests: %s\n', timestamp_str);
fprintf('Running sets: %s\n', strjoin(RUN_SETS, ', '));

if isempty(RUN_SETS)
    fprintf('No parameter sets selected to run. Please uncomment the sets you want to test.\n');
    return;
end

% Define all parameter sets (only the ones in RUN_SETS will be executed)
param_sets = {};

% 1. Different number of trees
if any(strcmp(RUN_SETS, 'TreesComparison'))
    param_sets{end+1} = struct('name', 'TreesComparison', ...
        'n_trees_list', [100, 200, 300, 500], ...
        'n_extracted_features_list', [25], ...
        'k_selected_features_list', [20], ...
        'mu_list', [40], ...
        'lambda_list', [120], ...
        'test_ratio_list', [0.2]);
end

% 2. Extracted features
if any(strcmp(RUN_SETS, 'ExtractedFeatures'))
    param_sets{end+1} = struct(...
        'name', 'ExtractedFeatures', ...
        'n_trees_list', [300], ...
        'n_extracted_features_list', [20, 25, 30, 35], ...
        'k_selected_features_list', [20], ...
        'mu_list', [40], ...
        'lambda_list', [120], ...
        'test_ratio_list', [0.2]);
end

% 3. Selected features
if any(strcmp(RUN_SETS, 'SelectedFeatures'))
    param_sets{end+1} = struct(...
        'name', 'SelectedFeatures', ...
        'n_trees_list', [300], ...
        'n_extracted_features_list', [25], ...
        'k_selected_features_list', [15, 20, 25, 30], ...
        'mu_list', [40], ...
        'lambda_list', [120], ...
        'test_ratio_list', [0.2]);
end

% 4. ES population sizes
if any(strcmp(RUN_SETS, 'PopulationSizes'))
    param_sets{end+1} = struct(...
        'name', 'PopulationSizes', ...
        'n_trees_list', [300], ...
        'n_extracted_features_list', [25], ...
        'k_selected_features_list', [20], ...
        'mu_list', [10, 15, 20, 25, 30], ...
        'lambda_list', [30, 45, 60, 75, 90], ... % 3 * mu_list
        'test_ratio_list', [0.2]);
end

% 5. ES offspring ratio
if any(strcmp(RUN_SETS, 'OffspringRatio'))
    param_sets{end+1} = struct(...
        'name', 'OffspringRatio', ...
        'n_trees_list', [300], ...
        'n_extracted_features_list', [25], ...
        'k_selected_features_list', [20], ...
        'lambda_list', [40, 60, 80, 100, 120], ...
        'mu_list', [13, 20, 27, 33, 40], ... % lambda_list / 3
        'test_ratio_list', [0.2]);
end

% 6. Test ratio comparison
if any(strcmp(RUN_SETS, 'TestRatioComparison'))
    param_sets{end+1} = struct(...
        'name', 'TestRatioComparison', ...
        'n_trees_list', [300], ...
        'n_extracted_features_list', [25], ...
        'k_selected_features_list', [20], ...
        'mu_list', [40], ...
        'lambda_list', [120], ...
        'test_ratio_list', [0.2, 0.25, 0.3, 0.4, 0.5]);
end

fixedParams = struct();
fixedParams.n_fuzzy_terms = 3;
fixedParams.T_max = 150;
fixedParams.k_folds_cv = 5;

%------------------------------
project_root = fullfile(fileparts(mfilename('fullpath')), '..');
project_root = fullfile(project_root);

addpath(fullfile(project_root, 'functions'));
addpath(fullfile(project_root, 'functions', 'functions_ES'));
addpath(fullfile(project_root, 'opened_nose'));
addpath(fullfile(project_root, 'closed_nose'));
addpath(fullfile(project_root, 'src'));
%------------------------------

% Opened nose
opened_nose_folder = fullfile(project_root, 'opened_nose');

HIO_opened = load(fullfile(opened_nose_folder, 'HIO.mat'));
data_HIO_opened = struct2array(HIO_opened);
KK_opened = load(fullfile(opened_nose_folder, 'KK.mat'));
data_KK_opened = struct2array(KK_opened);
KUA_opened = load(fullfile(opened_nose_folder, 'KUA.mat'));
data_KUA_opened = struct2array(KUA_opened);
SK_T_opened = load(fullfile(opened_nose_folder, 'SK_T.mat'));
data_SK_T_opened = struct2array(SK_T_opened);
SM_opened = load(fullfile(opened_nose_folder, 'SM.mat'));
data_SM_opened = struct2array(SM_opened);
opened_names = {'HIO', 'KK', 'KUA', 'SK_T', 'SM'};

% Closed nose
closed_nose_folder = fullfile(project_root, 'closed_nose');

AC_closed = load(fullfile(closed_nose_folder, 'AC.mat'));
data_AC_closed = struct2array(AC_closed);
BA_closed = load(fullfile(closed_nose_folder, 'BA.mat'));
data_BA_closed = struct2array(BA_closed);
BO_closed = load(fullfile(closed_nose_folder, 'BO.mat'));
data_BO_closed = struct2array(BO_closed);
CMY_closed = load(fullfile(closed_nose_folder, 'CMY.mat'));
data_CMY_closed = struct2array(CMY_closed);
JD_closed = load(fullfile(closed_nose_folder, 'JD.mat'));
data_JD_closed = struct2array(JD_closed);
closed_names = {'AC', 'BA', 'BO', 'CMY', 'JD'};

% Channel names
channel_names = load(fullfile(project_root, 'Channel_names.mat'));
channels = struct2array(channel_names);

% Define Variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data_AC_closed, 1);
n_time = size(data_AC_closed, 2);
n_channels = size(data_AC_closed, 3);
trial_duration_sec = n_time / sampling_rate;
total_duration_sec = n_trials * trial_duration_sec;

% Data Preprocessing
fprintf('\n   Data preprocessing: \n');

data_closed_combined = cat(4, data_AC_closed, data_BA_closed, data_BO_closed, data_CMY_closed, data_JD_closed);
data_opened_combined = cat(4, data_HIO_opened, data_KK_opened, data_KUA_opened, data_SK_T_opened, data_SM_opened);

EEG_data_closed_reshaped = cell(5,1);
EEG_data_opened_reshaped = cell(5,1);

for i = 1:5
    data_closed_permuted = permute(data_closed_combined(:,:,:,i), [3,2,1]);
    data_opened_permuted = permute(data_opened_combined(:,:,:,i), [3,2,1]);
   
    EEG_data_closed_reshaped{i} = reshape(data_closed_permuted, n_channels, []);
    EEG_data_opened_reshaped{i} = reshape(data_opened_permuted, n_channels, []);
end

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

clean_eeg_closed = zeros(size(EEG_filtered_closed));
clean_eeg_opened = zeros(size(EEG_filtered_opened));

fprintf('Running ICA.\n');
for i = 1:5
    clean_eeg_closed(:,:,:,i) = runICASubject(EEG_filtered_closed(:,:,:,i), 4, sampling_rate, closed_names{i}, 'closed', []);
    clean_eeg_opened(:,:,:,i) = runICASubject(EEG_filtered_opened(:,:,:,i), 4, sampling_rate, opened_names{i}, 'opened', []);
end

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

feature_matrix_normalized = (all_features_combined - mean(all_features_combined)) ./ std(all_features_combined);

%% Params testing
if isempty(gcp('nocreate'))
    parpool('local');
end

fprintf('\n   Testing parameters:\n');
fprintf('Total parameter sets: %d\n', length(param_sets));
fprintf('Runs this batch: %d (runs %d-%d out of %d total)\n', length(runs_this_batch), start_run, end_run, TOTAL_RUNS);

all_results = {};
overall_best_f1 = 0;
overall_best_config = [];
overall_best_set_name = '';

total_start_time = tic;

for set_idx = 1:length(param_sets)
    current_set = param_sets{set_idx};
   
    fprintf('\n=== Parameter set %d/%d: %s (Batch %d) ===\n', set_idx, length(param_sets), current_set.name, CURRENT_BATCH);
   
    % Total configs
    n_trees_list = current_set.n_trees_list;
    n_extracted_features_list = current_set.n_extracted_features_list;
    k_selected_features_list = current_set.k_selected_features_list;
    lambda_list = current_set.lambda_list;
    mu_list = current_set.mu_list;
    test_ratio_list = current_set.test_ratio_list;
   
    total_configs = length(n_trees_list) * length(n_extracted_features_list) * ...
                   length(k_selected_features_list) * length(lambda_list) * ...
                   length(mu_list) * length(test_ratio_list);
   
    fprintf('Configurations in this set: %d\n', total_configs);
   
    results_summary = [];
    config_idx = 0;
    set_start_time = tic;
   
    for n_trees = n_trees_list
        for n_extracted_features = n_extracted_features_list
            for k_selected_features = k_selected_features_list
                for lambda_numbers = lambda_list
                    for mu_numbers = mu_list
                        for test_ratio = test_ratio_list
                            config_idx = config_idx + 1;
                           
                            fprintf('\n Config %d/%d: T=%d, E=%d, S=%d, μ=%d, λ=%d, TestRatio=%.2f (Batch %d)\n', ...
                                config_idx, total_configs, n_trees, n_extracted_features, ...
                                k_selected_features, mu_numbers, lambda_numbers, test_ratio, CURRENT_BATCH);
                            
                            % Results storage - only for the runs in this batch
                            train_f1_runs = zeros(length(runs_this_batch), 1);
                            test_f1_runs = zeros(length(runs_this_batch), 1);
                            train_accuracy_runs = zeros(length(runs_this_batch), 1);
                            test_accuracy_runs = zeros(length(runs_this_batch), 1);
                            best_fitness_runs = zeros(length(runs_this_batch), 1);
                            test_polygon_area_runs = zeros(length(runs_this_batch), 1);
                           
                            config_start_time = tic;
                           
                            % Only run the specified batch of runs
                            for run_idx = 1:length(runs_this_batch)
                                actual_run = runs_this_batch(run_idx);
                                fprintf('  Run %d/%d (global run %d)...', run_idx, length(runs_this_batch), actual_run);
                               
                                % Set seed based on actual run number for consistency
                                rng(actual_run * 1000 + config_idx);
                               
                                unique_classes = unique(labels);
                                train_idx_global = [];
                                test_idx_global = [];
                               
                                for class_label = unique_classes'
                                    class_indices = find(labels == class_label);
                                    n_class_samples = length(class_indices);
                                    shuffled_class_idx = class_indices(randperm(n_class_samples));
                                    n_test_class = round(n_class_samples * test_ratio);
                                   
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
                               
                                % Calculate metrics
                                train_f1_runs(run_idx) = calculateF1Score(labels_train, y_pred_train_cv);
                                test_f1_runs(run_idx) = calculateF1Score(labels_test, y_pred_test);
                                train_accuracy_runs(run_idx) = sum(labels_train == y_pred_train_cv) / length(labels_train);
                                test_accuracy_runs(run_idx) = sum(labels_test == y_pred_test) / length(labels_test);
                                best_fitness_runs(run_idx) = max(fitness_history);
                               
                                temp_metrics = polygonareametric(labels_test, y_pred_test);
                                test_polygon_area_runs(run_idx) = temp_metrics.PA;
                               
                                fprintf(' F1=%.3f\n', test_f1_runs(run_idx));
                            end
                           
                            % Store configuration results for this batch
                            config_results = struct();
                            config_results.n_trees = n_trees;
                            config_results.n_extracted_features = n_extracted_features;
                            config_results.k_selected_features = k_selected_features;
                            config_results.lambda_numbers = lambda_numbers;
                            config_results.mu_numbers = mu_numbers;
                            config_results.test_ratio = test_ratio;
                            
                            % Store batch information
                            config_results.batch_number = CURRENT_BATCH;
                            config_results.runs_in_batch = runs_this_batch;
                            config_results.total_runs_planned = TOTAL_RUNS;
                           
                            % Store means and stds for this batch only
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
                           
                            % Store all individual run results with their global run numbers
                            config_results.all_runs = struct();
                            config_results.all_runs.train_f1 = train_f1_runs;
                            config_results.all_runs.test_f1 = test_f1_runs;
                            config_results.all_runs.train_accuracy = train_accuracy_runs;
                            config_results.all_runs.test_accuracy = test_accuracy_runs;
                            config_results.all_runs.best_fitness = best_fitness_runs;
                            config_results.all_runs.test_polygon_area = test_polygon_area_runs;
                            config_results.all_runs.global_run_numbers = runs_this_batch;
                           
                            results_summary = [results_summary; config_results];
                           
                            config_time = toc(config_start_time);
                            fprintf('  Config completed in %.1f min. Test F1: %.4f±%.4f (batch %d only)\n', ...
                                config_time/60, config_results.test_f1_mean, config_results.test_f1_std, CURRENT_BATCH);
                           
                            if config_results.test_f1_mean > overall_best_f1
                                overall_best_f1 = config_results.test_f1_mean;
                                overall_best_config = config_results;
                                overall_best_set_name = current_set.name;
                            end
                        end % test_ratio loop
                    end
                end
            end
        end
    end
   
    % Store results for this parameter set and batch
    set_results = struct();
    set_results.name = current_set.name;
    set_results.batch_number = CURRENT_BATCH;
    set_results.configs = results_summary;
    if ~isempty(results_summary)
        set_results.best_config = results_summary(find([results_summary.test_f1_mean] == max([results_summary.test_f1_mean]), 1));
    else
        set_results.best_config = [];
    end
    all_results{set_idx} = set_results;
   
    set_time = toc(set_start_time);
    fprintf('\n=== %s Batch %d Completed in %.1f hours ===\n', current_set.name, CURRENT_BATCH, set_time/3600);
    if ~isempty(results_summary)
        fprintf('Best config in batch: F1=%.4f±%.4f\n', set_results.best_config.test_f1_mean, set_results.best_config.test_f1_std);
    end
   
    % Save batch results
    save(sprintf('results_%s_%s_%s.mat', current_set.name, batch_str, timestamp_str), ...
     'set_results', 'fixedParams', 'TOTAL_RUNS', 'CURRENT_BATCH', 'runs_this_batch', 'timestamp_str');
    
    fprintf('Batch results saved for %s\n', current_set.name);
end

total_time = toc(total_start_time);

% Final Results Summary for this batch
fprintf('\n\n=== BATCH %d RESULTS ===\n', CURRENT_BATCH);
fprintf('Batch execution time: %.1f hours\n', total_time/3600);
fprintf('Parameter sets completed: %d\n', length(param_sets));
fprintf('Runs completed: %d-%d out of %d total\n', start_run, end_run, TOTAL_RUNS);

% Save batch summary
save(sprintf('batch_%d_summary_%s.mat', CURRENT_BATCH, timestamp_str), ...
     'all_results', 'CURRENT_BATCH', 'runs_this_batch', 'TOTAL_RUNS', ...
     'fixedParams', 'timestamp_str', 'total_time');

fprintf('\n=== NEXT STEPS ===\n');
if end_run < TOTAL_RUNS
    next_batch = CURRENT_BATCH + 1;
    next_start = end_run + 1;
    next_end = min(next_start + RUNS_PER_BATCH - 1, TOTAL_RUNS);
    fprintf('For next run:\n');
    fprintf('1. Change CURRENT_BATCH to %d\n', next_batch);
    fprintf('2. This will run runs %d-%d\n', next_start, next_end);
    fprintf('3. After all batches, use the concatenation script\n');
else
    fprintf('All batches complete! Use concatenation script to combine results.\n');
end

fprintf('Batch %d completed and saved!\n', CURRENT_BATCH);