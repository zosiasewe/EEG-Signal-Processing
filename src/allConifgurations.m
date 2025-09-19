clc
clear
close all

N_RUNS = 20;

timestamp_str = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
fprintf('Rozpoczynam testy parametrów: %s\n', timestamp_str);
param_sets = {};

% 1.Different number of trees
param_sets{end+1} = struct('name', 'TreesComparison', ...
    'n_trees_list', [100, 200, 300, 500], ...
    'n_extracted_features_list', [25], ...
    'k_selected_features_list', [20], ...
    'mu_list', [40], ...
    'lambda_list', [120], ...
    'test_ratio_list', [0.2]);

% 2.Extracted features
param_sets{end+1} = struct(...
    'name', 'ExtractedFeatures', ...
    'n_trees_list', [300], ...
    'n_extracted_features_list', [20, 25, 30, 35], ...
    'k_selected_features_list', [20], ...
    'mu_list', [40], ...
    'lambda_list', [120], ...
    'test_ratio_list', [0.2]);

% 3.Selected features
param_sets{end+1} = struct(...
    'name', 'SelectedFeatures', ...
    'n_trees_list', [300], ...
    'n_extracted_features_list', [25], ...
    'k_selected_features_list', [15, 20, 25, 30], ...
    'mu_list', [40], ...
    'lambda_list', [120], ...
    'test_ratio_list', [0.2]);

% 4.ES population sizes
param_sets{end+1} = struct(...
    'name', 'PopulationSizes', ...
    'n_trees_list', [300], ...
    'n_extracted_features_list', [25], ...
    'k_selected_features_list', [20], ...
    'mu_list', [10, 15, 20, 25, 30], ...
    'lambda_list', [30, 45, 60, 75, 90], ... % 3 * mu_list
    'test_ratio_list', [0.2]);

% 5.ES offspring ratio
param_sets{end+1} = struct(...
    'name', 'OffspringRatio', ...
    'n_trees_list', [300], ...
    'n_extracted_features_list', [25], ...
    'k_selected_features_list', [20], ...
    'lambda_list', [40, 60, 80, 100, 120], ...
    'mu_list', [13, 20, 27, 33, 40], ... % lambda_list / 3
    'test_ratio_list', [0.2]);

% 6. Test ratio comparison
param_sets{end+1} = struct(...
    'name', 'TestRatioComparison', ...
    'n_trees_list', [300], ...
    'n_extracted_features_list', [25], ...
    'k_selected_features_list', [20], ...
    'mu_list', [40], ...
    'lambda_list', [120], ...
    'test_ratio_list', [0.2, 0.25, 0.3, 0.4, 0.5]);

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
fprintf('Runs per configuration: %d\n', N_RUNS);

all_results = {};
overall_best_f1 = 0;
overall_best_config = [];
overall_best_set_name = '';

total_start_time = tic;

for set_idx = 1:length(param_sets)
    current_set = param_sets{set_idx};
   
    fprintf('\n   Parameters set %d/%d: %s ===\n', set_idx, length(param_sets), current_set.name);
   
    % Total configs - NOW INCLUDING TEST RATIO
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
                           
                            fprintf('\n Config %d/%d: T=%d, E=%d, S=%d, ?=%d, ?=%d, TestRatio=%.2f \n', ...
                                config_idx, total_configs, n_trees, n_extracted_features, ...
                                k_selected_features, mu_numbers, lambda_numbers, test_ratio);
                            
                            % ZAPISYWANIE DODATKWE
                            if mod(config_idx, 5) == 0  % co 5 konfiguracji
                                temp_results = results_summary;
                                save(sprintf('temp_%s_config_%d_%s.mat', current_set.name, config_idx, timestamp_str), ...
                                     'temp_results', 'current_set');
                                fprintf('  Temporary save: config %d/%d\n', config_idx, total_configs);
                            end

                            % results storage
                            train_f1_runs = zeros(N_RUNS, 1);
                            test_f1_runs = zeros(N_RUNS, 1);
                            train_accuracy_runs = zeros(N_RUNS, 1);
                            test_accuracy_runs = zeros(N_RUNS, 1);
                            best_fitness_runs = zeros(N_RUNS, 1);
                            test_polygon_area_runs = zeros(N_RUNS, 1);
                           
                            config_start_time = tic;
                           
                            for run = 1:N_RUNS
                                if mod(run, 10) == 0 || run == 1
                                    fprintf('  Run %d/%d...', run, N_RUNS);
                                end
                               
                                rng('shuffle');
                               
                                unique_classes = unique(labels);
                                train_idx_global = [];
                                test_idx_global = [];
                               
                                for class_label = unique_classes'
                                    class_indices = find(labels == class_label);
                                    n_class_samples = length(class_indices);
                                    shuffled_class_idx = class_indices(randperm(n_class_samples));
                                    n_test_class = round(n_class_samples * test_ratio); % USING VARIABLE
                                   
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
                                train_f1_runs(run) = calculateF1Score(labels_train, y_pred_train_cv);
                                test_f1_runs(run) = calculateF1Score(labels_test, y_pred_test);
                                train_accuracy_runs(run) = sum(labels_train == y_pred_train_cv) / length(labels_train);
                                test_accuracy_runs(run) = sum(labels_test == y_pred_test) / length(labels_test);
                                best_fitness_runs(run) = max(fitness_history);
                               
                                temp_metrics = polygonareametric(labels_test, y_pred_test);
                                test_polygon_area_runs(run) = temp_metrics.PA;
                               
                                if mod(run, 10) == 0 || run == 1
                                    fprintf(' F1=%.3f\n', test_f1_runs(run));
                                end
                            end
                           
                            % Store configuration results
                            config_results = struct();
                            config_results.n_trees = n_trees;
                            config_results.n_extracted_features = n_extracted_features;
                            config_results.k_selected_features = k_selected_features;
                            config_results.lambda_numbers = lambda_numbers;
                            config_results.mu_numbers = mu_numbers;
                            config_results.test_ratio = test_ratio; % NEW FIELD
                           
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
                           
                            config_results.all_runs = struct();
                            config_results.all_runs.train_f1 = train_f1_runs;
                            config_results.all_runs.test_f1 = test_f1_runs;
                            config_results.all_runs.train_accuracy = train_accuracy_runs;
                            config_results.all_runs.test_accuracy = test_accuracy_runs;
                            config_results.all_runs.best_fitness = best_fitness_runs;
                            config_results.all_runs.test_polygon_area = test_polygon_area_runs;
                           
                            results_summary = [results_summary; config_results];
                           
                            config_time = toc(config_start_time);
                            fprintf('  Config completed in %.1f min. Test F1: %.4f±%.4f, Acc: %.4f±%.4f\n', ...
                                config_time/60, config_results.test_f1_mean, config_results.test_f1_std, ...
                                config_results.test_accuracy_mean, config_results.test_accuracy_std);
                           
                            if config_results.test_f1_mean > overall_best_f1 %best
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
   
    % Storing results
    set_results = struct();
    set_results.name = current_set.name;
    set_results.configs = results_summary;
    set_results.best_config = results_summary(find([results_summary.test_f1_mean] == max([results_summary.test_f1_mean]), 1));
    all_results{set_idx} = set_results;
   
    set_time = toc(set_start_time);
    fprintf('\n    %s Completed in %.1f hours \n', current_set.name, set_time/3600);
    fprintf('Best config in set: F1=%.4f±%.4f\n', set_results.best_config.test_f1_mean, set_results.best_config.test_f1_std);
   
    save(sprintf('results_%s_%s.mat', current_set.name, timestamp_str), ...
     'set_results', 'fixedParams', 'N_RUNS', 'timestamp_str');
end
total_time = toc(total_start_time);

% ------------------------------------
fprintf('\n\n   Final Results \n');
fprintf('Total execution time: %.1f hours\n', total_time/3600);
fprintf('Total parameter sets tested: %d\n', length(param_sets));

fprintf('\nResults by parameter set:\n');
fprintf('%-20s %-15s %-15s %-15s %-15s\n', 'Set Name', 'Best Test F1', 'Best Test Acc', 'Configs', 'Time/Config');
fprintf('%s\n', repmat('-', 1, 80));

for i = 1:length(all_results)
    set_data = all_results{i};
    n_configs = length(set_data.configs);
    fprintf('%-20s %-15s %-15s %-15d %-15s\n', ...
        set_data.name, ...
        sprintf('%.4f+/-%.4f', set_data.best_config.test_f1_mean, set_data.best_config.test_f1_std), ...
        sprintf('%.4f+/-%.4f', set_data.best_config.test_accuracy_mean, set_data.best_config.test_accuracy_std), ...
        n_configs, ...
        sprintf('%.1f min', (total_time/3600*60)/(sum(cellfun(@(x) length(x.configs), all_results)))));
end

fprintf('\n   Best Configuration   \n');
fprintf('Parameter Set: %s\n', overall_best_set_name);
fprintf('Trees: %d, Extracted Features: %d, Selected Features: %d\n', ...
    overall_best_config.n_trees, overall_best_config.n_extracted_features, overall_best_config.k_selected_features);
fprintf('Parents: %d, Offspring: %d, Test Ratio: %.2f\n', ...
    overall_best_config.mu_numbers, overall_best_config.lambda_numbers, overall_best_config.test_ratio);
fprintf('Test F1: %.4f ± %.4f\n', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf('Test Accuracy: %.4f ± %.4f\n', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf('Test Polygon Area: %.4f ± %.4f\n', overall_best_config.test_polygon_area_mean, overall_best_config.test_polygon_area_std);

% Save all
save(sprintf('all_parameter_results_%s.mat', timestamp_str), ...
     'all_results', 'overall_best_config', 'overall_best_set_name', ...
     'fixedParams', 'N_RUNS', 'timestamp_str', 'total_time');

backup_filename = sprintf('backup_results_%s.mat', timestamp_str);
save(backup_filename, 'all_results', 'overall_best_config');
fprintf('Backup saved as: %s\n', backup_filename);

fprintf('\n Completed. \n');


% ----------------------------------------------------
%% Plots and Summary

if isempty(all_results)
    fprintf('Brak wyników do plotowania!\n');
    return;
end

fprintf('\n   Creating plots with timestamp: %s\n', timestamp_str);

colors = {'b','r','g','m','c','k'};
set_names = cell(length(all_results),1);
for i = 1:length(all_results)
    set_names{i} = all_results{i}.name;
end

% --------------------------
% Plot 1: Test F1 Score Across All Parameter Sets
fig1 = figure('Name','Test F1 Score Across Parameter Sets','Position',[50,50,1000,600]);
hold on;
x_offset = 0;
legend_entries = {};
for i = 1:length(all_results)
    set_data = all_results{i};
    test_f1_means = [set_data.configs.test_f1_mean];
    test_f1_stds = [set_data.configs.test_f1_std];
    x_pos = x_offset + (1:length(test_f1_means));
    errorbar(x_pos, test_f1_means, test_f1_stds,'Color',colors{mod(i-1,length(colors))+1},...
        'LineWidth',2,'Marker','o','MarkerSize',4);
    legend_entries{i} = set_data.name;
    x_offset = x_offset + length(test_f1_means) + 1;
end
xlabel('Configuration Index'); ylabel('Test F1 Score');
title('Test F1 Score Across All Parameter Sets');
legend(legend_entries,'Location','best','FontSize',8); grid on;
savefig(fig1,sprintf('plot1_testF1_%s.fig', timestamp_str)); 
print(fig1,sprintf('plot1_testF1_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 2: Test Accuracy Across Parameter Sets
fig2 = figure('Name','Test Accuracy Across Parameter Sets','Position',[50,50,1000,600]);
hold on; x_offset = 0;
for i = 1:length(all_results)
    set_data = all_results{i};
    test_acc_means = [set_data.configs.test_accuracy_mean];
    test_acc_stds = [set_data.configs.test_accuracy_std];
    x_pos = x_offset + (1:length(test_acc_means));
    errorbar(x_pos,test_acc_means,test_acc_stds,'Color',colors{mod(i-1,length(colors))+1},...
        'LineWidth',2,'Marker','s','MarkerSize',4);
    x_offset = x_offset + length(test_acc_means) + 1;
end
xlabel('Configuration Index'); ylabel('Test Accuracy');
title('Test Accuracy Across All Parameter Sets'); grid on;
savefig(fig2,sprintf('plot2_testAccuracy_%s.fig', timestamp_str)); 
print(fig2,sprintf('plot2_testAccuracy_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 3: Best F1 Performance by Parameter Set
fig3 = figure('Name','Best F1 by Parameter Set','Position',[50,50,1000,600]);
best_f1_scores = zeros(length(all_results),1);
best_f1_stds = zeros(length(all_results),1);
for i = 1:length(all_results)
    set_data = all_results{i};
    best_f1_scores(i) = set_data.best_config.test_f1_mean;
    best_f1_stds(i) = set_data.best_config.test_f1_std;
end
bar(1:length(best_f1_scores),best_f1_scores,'FaceColor',[0.3 0.7 0.9]); hold on;
errorbar(1:length(best_f1_scores),best_f1_scores,best_f1_stds,'k.','LineWidth',2);
xlabel('Parameter Set'); ylabel('Best Test F1 Score');
title('Best F1 Performance by Parameter Set');
set(gca,'XTick',1:length(set_names),'XTickLabel',set_names); xtickangle(45); grid on;
savefig(fig3,sprintf('plot3_bestF1_%s.fig', timestamp_str)); 
print(fig3,sprintf('plot3_bestF1_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 4: Box plots of F1 score distributions by parameter set
% Plot 4: Box plots of F1 score distributions by parameter set
fig4 = figure('Name','F1 Score Distribution by Parameter Set','Position',[50,50,1200,700]);
all_f1_data = []; group_labels = {};
max_points_per_set = 200; % ogranicz liczb? punktów

for i = 1:length(all_results)
    set_data = all_results{i};
    all_runs_in_set = [];
    for j = 1:length(set_data.configs)
        all_runs_in_set = [all_runs_in_set; set_data.configs(j).all_runs.test_f1];
    end
    
    % Ogranicz liczb? punktów je?li za du?o
    if length(all_runs_in_set) > max_points_per_set
        idx = randperm(length(all_runs_in_set), max_points_per_set);
        all_runs_in_set = all_runs_in_set(idx);
    end
    
    all_f1_data = [all_f1_data; all_runs_in_set];
    group_labels = [group_labels; repmat({set_data.name},length(all_runs_in_set),1)];
end
boxplot(all_f1_data, group_labels,'Whisker',1.5,'Symbol','r+');
ylabel('Test F1 Score'); title('F1 Score Distributions by Parameter Set');
xtickangle(30); grid on;
savefig(fig4,sprintf('plot4_F1_boxplot_%s.fig', timestamp_str)); 
print(fig4,sprintf('plot4_F1_boxplot_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 5: Polygon Area Metric
fig5 = figure('Name','Polygon Area Metric','Position',[50,50,1000,600]);
hold on; x_offset=0;
for i=1:length(all_results)
    set_data = all_results{i};
    poly_means = [set_data.configs.test_polygon_area_mean];
    poly_stds = [set_data.configs.test_polygon_area_std];
    x_pos = x_offset + (1:length(poly_means));
    errorbar(x_pos,poly_means,poly_stds,'Color',colors{mod(i-1,length(colors))+1},...
        'LineWidth',2,'Marker','^','MarkerSize',4);
    x_offset = x_offset + length(poly_means) + 1;
end
xlabel('Configuration Index'); ylabel('Test Polygon Area');
title('Polygon Area Metric Across Parameter Sets'); grid on;
savefig(fig5,sprintf('plot5_polygonArea_%s.fig', timestamp_str)); 
print(fig5,sprintf('plot5_polygonArea_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 6: Performance vs Computational Cost
fig6 = figure('Name','Performance vs Computational Cost','Position',[50,50,1000,600]);
n_configs_per_set = cellfun(@(x) length(x.configs), all_results);
computational_cost = n_configs_per_set * N_RUNS;
scatter(computational_cost,best_f1_scores,100,1:length(all_results),'filled'); colorbar;
xlabel('Computational Cost (Configs × Runs)'); ylabel('Best Test F1 Score');
title('Performance vs Computational Cost');
for i=1:length(all_results)
    text(computational_cost(i),best_f1_scores(i),sprintf('  %s',set_names{i}),'FontSize',8);
end
grid on;
savefig(fig6,sprintf('plot6_perf_vs_cost_%s.fig', timestamp_str)); 
print(fig6,sprintf('plot6_perf_vs_cost_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 7: Train vs Test F1 Comparison
fig7 = figure('Name','Train vs Test F1 Comparison','Position',[50,50,1000,600]);
all_train_f1 = []; all_test_f1 = [];
for i=1:length(all_results)
    set_data = all_results{i};
    all_train_f1 = [all_train_f1, [set_data.configs.train_f1_mean]];
    all_test_f1 = [all_test_f1, [set_data.configs.test_f1_mean]];
end
scatter(all_train_f1, all_test_f1,50,'filled');
xlabel('Train F1 Score'); ylabel('Test F1 Score'); title('Train vs Test F1 Performance');
min_val = min([all_train_f1, all_test_f1]); max_val = max([all_train_f1, all_test_f1]);
line([min_val max_val],[min_val max_val],'Color','r','LineStyle','--'); grid on;
savefig(fig7,sprintf('plot7_train_vs_test_%s.fig', timestamp_str)); 
print(fig7,sprintf('plot7_train_vs_test_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 8: Impact of Test Ratio
fig8 = figure('Name','Test Ratio Impact','Position',[50,50,1000,600]);
test_ratio_idx = find(strcmp({all_results{:}.name}, 'TestRatioComparison'));
if ~isempty(test_ratio_idx)
    test_ratio_data = all_results{test_ratio_idx};
    test_ratios = [test_ratio_data.configs.test_ratio];
    test_f1_means = [test_ratio_data.configs.test_f1_mean];
    test_f1_stds = [test_ratio_data.configs.test_f1_std];
    errorbar(test_ratios,test_f1_means,test_f1_stds,'bo-','LineWidth',2,'MarkerSize',8);
    xlabel('Test Ratio'); ylabel('Test F1 Score'); title('Impact of Train/Test Split Ratio'); grid on;
else
    set_means = cellfun(@(x) mean([x.configs.test_f1_mean]), all_results);
    set_stds = cellfun(@(x) std([x.configs.test_f1_mean]), all_results);
    bar(1:length(set_means),set_means,'FaceColor',[0.7 0.7 0.9]); hold on;
    errorbar(1:length(set_means),set_means,set_stds,'k.','LineWidth',2);
    xlabel('Parameter Set'); ylabel('Mean F1 Across Configs');
    title('Average Performance by Parameter Set'); set(gca,'XTick',1:length(set_names),'XTickLabel',set_names); xtickangle(45); grid on;
end
savefig(fig8,sprintf('plot8_testRatioImpact_%s.fig', timestamp_str)); 
print(fig8,sprintf('plot8_testRatioImpact_%s.png', timestamp_str),'-dpng','-r300');
% --------------------------
% Plot 9: F1 Score Variance by Parameter Set
fig9 = figure('Name','F1 Score Variance by Parameter Set','Position',[50,50,1000,600]);
set_variances = cellfun(@(x) std([x.configs.test_f1_mean]), all_results);
bar(1:length(set_variances),set_variances,'FaceColor',[0.9 0.5 0.5]);
xlabel('Parameter Set'); ylabel('F1 Score Variance'); title('Performance Variability by Parameter Set');
set(gca,'XTick',1:length(set_names),'XTickLabel',set_names); xtickangle(45); grid on;
savefig(fig9,sprintf('plot9_F1_variance_%s.fig', timestamp_str)); 
print(fig9,sprintf('plot9_F1_variance_%s.png', timestamp_str),'-dpng','-r300');


fprintf('\n   SUMMARY STATISTICS \n');
fprintf('Total configurations tested: %d\n', sum(cellfun(@(x) length(x.configs), all_results)));
fprintf('Total runs executed: %d\n', sum(cellfun(@(x) length(x.configs), all_results)) * N_RUNS);
fprintf('Average time per config: %.2f minutes\n', (total_time/60) / sum(cellfun(@(x) length(x.configs), all_results)));

fprintf('All 9 figures saved as .fig and .png files with timestamp: %s\n', timestamp_str);
