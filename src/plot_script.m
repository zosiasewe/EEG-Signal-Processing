clc
clear
close all

%% CONCATENATE BATCH RESULTS AND CREATE COMPREHENSIVE ANALYSIS
timestamp_str = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
fprintf('=== Batch Results Concatenation & Analysis ===\n');
fprintf('Timestamp: %s\n\n', timestamp_str);

%% 1. DEFINE WHICH SETS ARE BATCHED VS COMPLETE
batched_sets = {'PopulationSizes', 'OffspringRatio', 'TestRatioComparison'};
complete_sets = {'TreesComparison', 'ExtractedFeatures', 'SelectedFeatures'};

all_results = {};

%% 2. LOAD COMPLETE RESULTS (non-batched)
fprintf('Loading complete results...\n');
for i = 1:length(complete_sets)
    set_name = complete_sets{i};
    pattern = sprintf('results_%s_*.mat', set_name);
    files = dir(pattern);
    
    % Exclude batch files
    files = files(~contains({files.name}, 'batch'));
    
    if ~isempty(files)
        [~, idx] = max([files.datenum]);
        data = load(files(idx).name);
        all_results{end+1} = data.set_results;
        fprintf('  ✓ %s: %d configs, %d runs each\n', set_name, ...
            length(data.set_results.configs), data.N_RUNS);
    else
        fprintf('  ✗ WARNING: No file found for %s\n', set_name);
    end
end

%% 3. CONCATENATE BATCH RESULTS
fprintf('\nConcatenating batch results...\n');

for i = 1:length(batched_sets)
    set_name = batched_sets{i};
    pattern = sprintf('results_%s_batch*_*.mat', set_name);
    batch_files = dir(pattern);
    
    if isempty(batch_files)
        fprintf('  ✗ WARNING: No batch files found for %s\n', set_name);
        continue;
    end
    
    fprintf('  Processing %s: found %d batch files\n', set_name, length(batch_files));
    
    % Extract batch info from all files
    batch_info = struct('file', {}, 'batch_num', {}, 'timestamp', {});
    
    for j = 1:length(batch_files)
        fname = batch_files(j).name;
        batch_match = regexp(fname, 'batch(\d+)', 'tokens');
        if isempty(batch_match)
            continue;
        end
        batch_num = str2double(batch_match{1}{1});
        timestamp_match = regexp(fname, '\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', 'match');
        timestamp_key = timestamp_match{end};
        
        batch_info(end+1).file = fname;
        batch_info(end).batch_num = batch_num;
        batch_info(end).timestamp = timestamp_key;
    end
    
    % Sort by batch number (ignore timestamp - we want ALL batches together)
    [~, sort_idx] = sort([batch_info.batch_num]);
    batch_info = batch_info(sort_idx);
    
    fprintf('    Merging ALL %d batches together...\n', length(batch_info));
    
    % Load first batch as template
    data1 = load(batch_info(1).file);
    merged_results = data1.set_results;
    n_configs = length(merged_results.configs);
    
    % Initialize concatenated results
    for c = 1:n_configs
        merged_results.configs(c).all_runs.train_f1 = [];
        merged_results.configs(c).all_runs.test_f1 = [];
        merged_results.configs(c).all_runs.train_accuracy = [];
        merged_results.configs(c).all_runs.test_accuracy = [];
        merged_results.configs(c).all_runs.best_fitness = [];
        merged_results.configs(c).all_runs.test_polygon_area = [];
        merged_results.configs(c).all_runs.global_run_numbers = [];
    end
    
    % Concatenate ALL batches
    for b = 1:length(batch_info)
        data = load(batch_info(b).file);
        batch_data = data.set_results;
        
        batch_num = batch_data.configs(1).batch_number;
        runs_in_batch = batch_data.configs(1).runs_in_batch;
        
        fprintf('      Batch %d: runs %d-%d (timestamp: %s)\n', batch_num, ...
            min(runs_in_batch), max(runs_in_batch), batch_info(b).timestamp);
        
        for c = 1:n_configs
            merged_results.configs(c).all_runs.train_f1 = ...
                [merged_results.configs(c).all_runs.train_f1; batch_data.configs(c).all_runs.train_f1];
            merged_results.configs(c).all_runs.test_f1 = ...
                [merged_results.configs(c).all_runs.test_f1; batch_data.configs(c).all_runs.test_f1];
            merged_results.configs(c).all_runs.train_accuracy = ...
                [merged_results.configs(c).all_runs.train_accuracy; batch_data.configs(c).all_runs.train_accuracy];
            merged_results.configs(c).all_runs.test_accuracy = ...
                [merged_results.configs(c).all_runs.test_accuracy; batch_data.configs(c).all_runs.test_accuracy];
            merged_results.configs(c).all_runs.best_fitness = ...
                [merged_results.configs(c).all_runs.best_fitness; batch_data.configs(c).all_runs.best_fitness];
            merged_results.configs(c).all_runs.test_polygon_area = ...
                [merged_results.configs(c).all_runs.test_polygon_area; batch_data.configs(c).all_runs.test_polygon_area];
            merged_results.configs(c).all_runs.global_run_numbers = ...
                [merged_results.configs(c).all_runs.global_run_numbers; runs_in_batch(:)];
        end
    end
    
    % Recalculate statistics with ALL runs
    for c = 1:n_configs
        merged_results.configs(c).train_f1_mean = mean(merged_results.configs(c).all_runs.train_f1);
        merged_results.configs(c).train_f1_std = std(merged_results.configs(c).all_runs.train_f1);
        merged_results.configs(c).test_f1_mean = mean(merged_results.configs(c).all_runs.test_f1);
        merged_results.configs(c).test_f1_std = std(merged_results.configs(c).all_runs.test_f1);
        merged_results.configs(c).train_accuracy_mean = mean(merged_results.configs(c).all_runs.train_accuracy);
        merged_results.configs(c).train_accuracy_std = std(merged_results.configs(c).all_runs.train_accuracy);
        merged_results.configs(c).test_accuracy_mean = mean(merged_results.configs(c).all_runs.test_accuracy);
        merged_results.configs(c).test_accuracy_std = std(merged_results.configs(c).all_runs.test_accuracy);
        merged_results.configs(c).best_fitness_mean = mean(merged_results.configs(c).all_runs.best_fitness);
        merged_results.configs(c).best_fitness_std = std(merged_results.configs(c).all_runs.best_fitness);
        merged_results.configs(c).test_polygon_area_mean = mean(merged_results.configs(c).all_runs.test_polygon_area);
        merged_results.configs(c).test_polygon_area_std = std(merged_results.configs(c).all_runs.test_polygon_area);
        merged_results.configs(c).total_runs_completed = length(merged_results.configs(c).all_runs.test_f1);
    end
    
    % Find best config
    [~, best_idx] = max([merged_results.configs.test_f1_mean]);
    merged_results.best_config = merged_results.configs(best_idx);
    
    % Add ONCE to results
    all_results{end+1} = merged_results;
    
    fprintf('    ✓ Merged %s: %d configs, %d total runs each\n', set_name, ...
        n_configs, merged_results.configs(1).total_runs_completed);
    
    % Save merged results
    set_results = merged_results;
    save(sprintf('results_%s_MERGED_%s.mat', set_name, timestamp_str), 'set_results');
end

%% 4. VERIFY ALL RESULTS
fprintf('\n=== VERIFICATION ===\n');
total_configs = 0;
for i = 1:length(all_results)
    n_configs = length(all_results{i}.configs);
    
    % Check if total_runs_completed exists, otherwise count from all_runs
    if isfield(all_results{i}.configs(1), 'total_runs_completed')
        n_runs = all_results{i}.configs(1).total_runs_completed;
    else
        n_runs = length(all_results{i}.configs(1).all_runs.test_f1);
    end
    
    total_configs = total_configs + n_configs;
    fprintf('%s: %d configs × %d runs = %d total runs\n', ...
        all_results{i}.name, n_configs, n_runs, n_configs * n_runs);
end
fprintf('TOTAL: %d configurations\n', total_configs);
%% 4b. STANDARDIZE ALL RESULTS (add missing fields)
fprintf('\nStandardizing result structures...\n');
for i = 1:length(all_results)
    for c = 1:length(all_results{i}.configs)
        % Add total_runs_completed if missing
        if ~isfield(all_results{i}.configs(c), 'total_runs_completed')
            all_results{i}.configs(c).total_runs_completed = ...
                length(all_results{i}.configs(c).all_runs.test_f1);
        end
    end
end
fprintf('  ✓ All results standardized\n');
%% 5. FIND OVERALL BEST CONFIGURATION
fprintf('\n=== FINDING BEST CONFIGURATION ===\n');
overall_best_f1 = 0;
overall_best_config = [];
overall_best_set = '';

for i = 1:length(all_results)
    if all_results{i}.best_config.test_f1_mean > overall_best_f1
        overall_best_f1 = all_results{i}.best_config.test_f1_mean;
        overall_best_config = all_results{i}.best_config;
        overall_best_set = all_results{i}.name;
    end
end

fprintf('Best Configuration Found:\n');
fprintf('  Parameter Set: %s\n', overall_best_set);
fprintf('  Test F1: %.4f ± %.4f\n', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf('  Test Accuracy: %.4f ± %.4f\n', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf('  Configuration:\n');
fprintf('    Trees: %d\n', overall_best_config.n_trees);
fprintf('    Extracted Features: %d\n', overall_best_config.n_extracted_features);
fprintf('    Selected Features: %d\n', overall_best_config.k_selected_features);
fprintf('    μ (parents): %d\n', overall_best_config.mu_numbers);
fprintf('    λ (offspring): %d\n', overall_best_config.lambda_numbers);
fprintf('    Test Ratio: %.2f\n', overall_best_config.test_ratio);

%% 6b. ROBUST TOTAL RUNS (all configs across all sets)
total_runs_all = count_total_runs(all_results);

%% 7. CREATE COMPREHENSIVE TEXT SUMMARY
fprintf('\n=== CREATING TEXT SUMMARY ===\n');

fid = fopen(sprintf('Results_Summary_%s.txt', timestamp_str), 'w');

fprintf(fid, '============================================\n');
fprintf(fid, 'COMPREHENSIVE PARAMETER OPTIMIZATION RESULTS\n');
fprintf(fid, '============================================\n');
fprintf(fid, 'Generated: %s\n\n', timestamp_str);

fprintf(fid, 'OVERALL BEST CONFIGURATION:\n');
fprintf(fid, '  Parameter Set: %s\n', overall_best_set);
fprintf(fid, '  Test F1 Score: %.4f ± %.4f\n', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf(fid, '  Test Accuracy: %.4f ± %.4f\n', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf(fid, '  Train F1 Score: %.4f ± %.4f\n', overall_best_config.train_f1_mean, overall_best_config.train_f1_std);
fprintf(fid, '  Polygon Area: %.4f ± %.4f\n\n', overall_best_config.test_polygon_area_mean, overall_best_config.test_polygon_area_std);

fprintf(fid, '  Hyperparameters:\n');
fprintf(fid, '    - Number of Trees: %d\n', overall_best_config.n_trees);
fprintf(fid, '    - Extracted Features: %d\n', overall_best_config.n_extracted_features);
fprintf(fid, '    - Selected Features: %d\n', overall_best_config.k_selected_features);
fprintf(fid, '    - ES Parents (μ): %d\n', overall_best_config.mu_numbers);
fprintf(fid, '    - ES Offspring (λ): %d\n', overall_best_config.lambda_numbers);
fprintf(fid, '    - Test Ratio: %.2f\n\n', overall_best_config.test_ratio);

fprintf(fid, '============================================\n');
fprintf(fid, 'RESULTS BY PARAMETER SET:\n');
fprintf(fid, '============================================\n\n');

for i = 1:length(all_results)
    fprintf(fid, '%d. %s\n', i, all_results{i}.name);
    fprintf(fid, '   Configurations tested: %d\n', length(all_results{i}.configs));
    fprintf(fid, '   Runs per configuration: %d\n', all_results{i}.configs(1).total_runs_completed);
    fprintf(fid, '\n   Best Configuration:\n');
    best_cfg = all_results{i}.best_config;
    fprintf(fid, '     Test F1: %.4f ± %.4f\n', best_cfg.test_f1_mean, best_cfg.test_f1_std);
    fprintf(fid, '     Test Acc: %.4f ± %.4f\n', best_cfg.test_accuracy_mean, best_cfg.test_accuracy_std);
    fprintf(fid, '     Parameters: T=%d, E=%d, S=%d, μ=%d, λ=%d, TestRatio=%.2f\n', ...
        best_cfg.n_trees, best_cfg.n_extracted_features, best_cfg.k_selected_features, ...
        best_cfg.mu_numbers, best_cfg.lambda_numbers, best_cfg.test_ratio);
    
    fprintf(fid, '\n   All Configurations (F1 Mean ± Std):\n');
    for c = 1:length(all_results{i}.configs)
        cfg = all_results{i}.configs(c);
        fprintf(fid, '     Config %d: %.4f ± %.4f  [T=%d, E=%d, S=%d, μ=%d, λ=%d, TR=%.2f]\n', ...
            c, cfg.test_f1_mean, cfg.test_f1_std, cfg.n_trees, cfg.n_extracted_features, ...
            cfg.k_selected_features, cfg.mu_numbers, cfg.lambda_numbers, cfg.test_ratio);
    end
    fprintf(fid, '\n');
end

fprintf(fid, '============================================\n');
fprintf(fid, 'STATISTICAL SUMMARY:\n');
fprintf(fid, '============================================\n\n');

fprintf(fid, 'Total Configurations: %d\n', total_configs);
fprintf(fid, 'Total Experimental Runs: %.0f\n', total_runs_all);

all_f1_means = [];
for i = 1:length(all_results)
    all_f1_means = [all_f1_means; [all_results{i}.configs.test_f1_mean]'];
end

fprintf(fid, '\nF1 Score Statistics Across All Configurations:\n');
fprintf(fid, '  Mean: %.4f\n', mean(all_f1_means));
fprintf(fid, '  Std: %.4f\n', std(all_f1_means));
fprintf(fid, '  Min: %.4f\n', min(all_f1_means));
fprintf(fid, '  Max: %.4f\n', max(all_f1_means));
fprintf(fid, '  Median: %.4f\n', median(all_f1_means));

fclose(fid);

%% 8. CREATE LATEX TABLE FOR PUBLICATION
fprintf('\n=== CREATING LATEX TABLE ===\n');

fid_tex = fopen(sprintf('Results_Table_%s.tex', timestamp_str), 'w');

fprintf(fid_tex, '\\begin{table}[htbp]\n');
fprintf(fid_tex, '\\centering\n');
fprintf(fid_tex, '\\caption{Parameter Optimization Results - Best Configuration per Parameter Set}\n');
fprintf(fid_tex, '\\label{tab:results}\n');
fprintf(fid_tex, '\\begin{tabular}{lcccccc}\n');
fprintf(fid_tex, '\\hline\n');
fprintf(fid_tex, '\\textbf{Parameter Set} & \\textbf{Test F1} & \\textbf{Test Acc} & \\textbf{Train F1} & \\textbf{Configs} & \\textbf{Runs} \\\\\n');
fprintf(fid_tex, '\\hline\n');

for i = 1:length(all_results)
    best_cfg = all_results{i}.best_config;
    set_name_clean = strrep(all_results{i}.name, '_', '\\_');
    
    % Get number of runs (handle both old and new structure)
    if isfield(best_cfg, 'total_runs_completed')
        n_runs = best_cfg.total_runs_completed;
    else
        n_runs = length(best_cfg.all_runs.test_f1);
    end
    
    fprintf(fid_tex, '%s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & %d & %d \\\\\n', ...
        set_name_clean, ...
        best_cfg.test_f1_mean, best_cfg.test_f1_std, ...
        best_cfg.test_accuracy_mean, best_cfg.test_accuracy_std, ...
        best_cfg.train_f1_mean, best_cfg.train_f1_std, ...
        length(all_results{i}.configs), ...
        n_runs);
end

fprintf(fid_tex, '\\hline\n');
fprintf(fid_tex, '\\end{tabular}\n');
fprintf(fid_tex, '\\end{table}\n\n');

%% 9. CREATE CSV EXPORT FOR FURTHER ANALYSIS
fprintf('\n=== CREATING CSV EXPORTS ===\n');

% Export all configurations
csv_filename = sprintf('All_Configs_%s.csv', timestamp_str);
fid_csv = fopen(csv_filename, 'w');
fprintf(fid_csv, 'ParameterSet,ConfigID,Trees,ExtractedFeatures,SelectedFeatures,Mu,Lambda,TestRatio,TestF1Mean,TestF1Std,TestAccMean,TestAccStd,TrainF1Mean,TrainF1Std,PolygonAreaMean,PolygonAreaStd,NumRuns\n');

for i = 1:length(all_results)
    for c = 1:length(all_results{i}.configs)
        cfg = all_results{i}.configs(c);
        fprintf(fid_csv, '%s,%d,%d,%d,%d,%d,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n', ...
            all_results{i}.name, c, ...
            cfg.n_trees, cfg.n_extracted_features, cfg.k_selected_features, ...
            cfg.mu_numbers, cfg.lambda_numbers, cfg.test_ratio, ...
            cfg.test_f1_mean, cfg.test_f1_std, ...
            cfg.test_accuracy_mean, cfg.test_accuracy_std, ...
            cfg.train_f1_mean, cfg.train_f1_std, ...
            cfg.test_polygon_area_mean, cfg.test_polygon_area_std, ...
            cfg.total_runs_completed);
    end
end
fclose(fid_csv);
fprintf('  ✓ CSV exported: %s\n', csv_filename);

%% 10. SAVE FINAL WORKSPACE
save(sprintf('FINAL_ALL_RESULTS_%s.mat', timestamp_str), 'all_results', 'overall_best_config', 'overall_best_set');
fprintf('\n  ✓ Final workspace saved\n');

% Safety: recompute if empty or non-finite
if ~exist('total_runs_all','var') || isempty(total_runs_all) || ~isfinite(total_runs_all)
    total_runs_all = count_total_runs(all_results);
end
%% 11. FINAL SUMMARY TO CONSOLE
fprintf('\n');
fprintf('========================================\n');
fprintf('    ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('Generated files:\n');
fprintf('  • 4 publication-quality figures (PNG + FIG)\n');
fprintf('  • 1 comprehensive text summary\n');
fprintf('  • 1 LaTeX table file\n');
fprintf('  • 1 CSV export for all configurations\n');
fprintf('  • 1 merged MATLAB workspace\n');
fprintf('  • %d merged results files (for batched sets)\n', length(batched_sets));
fprintf('\nKey Findings:\n');
fprintf('  Best Parameter Set: %s\n', overall_best_set);
fprintf('  Best Test F1: %.4f ± %.4f\n', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf('  Best Test Accuracy: %.4f ± %.4f\n', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf('\nTotal Experimental Work:\n');
fprintf('  Configurations: %d\n', total_configs);
fprintf('  Total Runs: %.0f\n', total_runs_all);
fprintf('========================================\n');



%% ===================== NEW PUBLICATION FIGURES (PASTEL) =====================
fprintf('\n=== NEW PUBLICATION FIGURES (PASTEL) ===\n');

if ~exist('timestamp_str','var') || isempty(timestamp_str)
    timestamp_str = datestr(now,'yyyy_mm_dd_HH_MM_SS');
end

% ---------- GLOBAL LOOK ----------
set(0,'DefaultAxesFontSize',11);
set(0,'DefaultAxesLineWidth',1.1);
set(0,'DefaultLineLineWidth',1.6);
set(0,'DefaultFigureColor','w');

% ---------- PASTEL PALETTE ----------
colors_pastel = [
    0.69 0.84 0.96  % pastel blue
    0.99 0.79 0.67  % pastel orange
    0.99 0.90 0.65  % pastel yellow
    0.78 0.69 0.89  % pastel purple
    0.75 0.89 0.67  % pastel green
    0.70 0.88 0.90  % pastel cyan
];

% ---------- PREP COMMON DATA ----------
n_sets = numel(all_results);
set_names = cell(n_sets,1);
best_f1_means = zeros(n_sets,1);
best_f1_stds  = zeros(n_sets,1);
best_f1_ci    = zeros(n_sets,1);
all_data_by_set = cell(n_sets,1);

for i = 1:n_sets
    set_names{i}   = strrep(all_results{i}.name,'_',' ');
    best_f1_means(i) = all_results{i}.best_config.test_f1_mean;
    best_f1_stds(i)  = all_results{i}.best_config.test_f1_std;
    % n for CI comes from all runs of the best config
    n_runs_i = numel(all_results{i}.best_config.all_runs.test_f1);
    best_f1_ci(i) = 1.96 * best_f1_stds(i) / sqrt(max(1,n_runs_i));
    % gather all F1 scores across all configs/runs
    tmp = [];
    for c = 1:numel(all_results{i}.configs)
        tmp = [tmp; all_results{i}.configs(c).all_runs.test_f1];
    end
    all_data_by_set{i} = tmp;
end

% ======================================================================
% FIGURE A — Overview: Best (bar+CI) + All distributions (violins)
% ======================================================================
figA = figure('Position',[80 80 1400 650],'Color','w');

% A1: Best config by set (bar + 95% CI)
subplot(1,2,1); hold on;
b = bar(1:n_sets, best_f1_means, 0.7, 'FaceColor','flat');
for i = 1:n_sets
    b.CData(i,:) = colors_pastel(mod(i-1,size(colors_pastel,1))+1,:);
end
errorbar(1:n_sets, best_f1_means, best_f1_ci, 'k','LineStyle','none','LineWidth',1.4,'CapSize',8);
ylabel('Test F1 Score');
xlabel('Parameter Set');
set(gca,'FontSize',13);
title('Best Configuration Performance (95% CI)');
set(gca,'XTick',1:n_sets,'XTickLabel',set_names,'XTickLabelRotation',45);
grid on; box on;
ylim([min(best_f1_means-best_f1_ci)-0.01, max(best_f1_means+best_f1_ci)+0.01]);
[~,bestIdx] = max(best_f1_means);
text(bestIdx, best_f1_means(bestIdx) + 0.002, '★', ...
    'HorizontalAlignment','center','FontSize',12,'Color','k');

% A2: Violin distributions for all runs across all configs
subplot(1,2,2); hold on;
pos = 1:n_sets;
for i = 1:n_sets
    data = all_data_by_set{i};
    if numel(data) < 3, continue; end
    [f, xi] = ksdensity(data,'Bandwidth',0.005,'NumPoints',200);
    f = 0.35 * f / max(f);  % normalized half-width
    col = colors_pastel(mod(i-1,size(colors_pastel,1))+1,:);
    fill([pos(i)-f, fliplr(pos(i)+f)], [xi, fliplr(xi)], col, ...
        'FaceAlpha',0.6,'EdgeColor',col,'LineWidth',1.1);
    % median + IQR
    med = median(data); q25 = prctile(data,25); q75 = prctile(data,75);
    plot([pos(i)-0.33, pos(i)+0.33], [med med], 'k-','LineWidth',2.2);
    plot([pos(i) pos(i)], [q25 q75], 'k-','LineWidth',2);
end
ylabel('Test F1 Score'); xlabel('Parameter Set');
set(gca,'FontSize',13);
title('F1 Distribution Across All Configurations');
set(gca,'XTick',pos,'XTickLabel',set_names,'XTickLabelRotation',45);
grid on; box on;
ylim([0.76 0.94]);

saveas(figA, sprintf('PubFig_A_Overview_%s.png', timestamp_str));
savefig(figA, sprintf('PubFig_A_Overview_%s.fig', timestamp_str));
fprintf('  ✓ PubFig A saved\n');

% ======================================================================
% FIGURE B — Effect size vs baseline + Consistency (CV%%)
% ======================================================================
figB = figure('Position',[80 80 1400 600],'Color','w');

% Choose baseline = first set (TreesComparison in your runs)
baseline = all_data_by_set{1};
effect_sizes = zeros(n_sets,1);
for i = 1:n_sets
    pooled = sqrt( (std(baseline)^2 + std(all_data_by_set{i})^2) / 2 );
    effect_sizes(i) = (mean(all_data_by_set{i}) - mean(baseline)) / max(eps,pooled);
end

% B1: Cohen's d (horizontal bars)
subplot(1,2,1); hold on;
bh = barh(1:n_sets, effect_sizes, 0.7, 'FaceColor','flat');
for i = 1:n_sets, bh.CData(i,:) = colors_pastel(mod(i-1,size(colors_pastel,1))+1,:); end
plot([0.2 0.2], [0 n_sets+1], 'k--','LineWidth',1);     % small
plot([0.5 0.5], [0 n_sets+1], 'k:','LineWidth',1.2);    % medium
plot([0.8 0.8], [0 n_sets+1], 'k-.','LineWidth',1.2);   % large
xlabel('Effect Size (Cohen''s d)'); ylabel('Parameter Set');
set(gca,'YTick',1:n_sets,'YTickLabel',set_names);
xline(0,'k:','LineWidth',1);      % zero reference
set(gca,'FontSize',13);
title('Effect Size vs. Baseline'); grid on; box on;
xlim([min(-0.6, min(effect_sizes)-0.05) 0.8]);

% B2: Consistency via coefficient of variation (%)
subplot(1,2,2); hold on;
cv_pct = zeros(n_sets,1);
for i = 1:n_sets
    cv_pct(i) = std(all_data_by_set{i}) / max(mean(all_data_by_set{i}),eps) * 100;
end
bb = bar(1:n_sets, cv_pct, 0.7, 'FaceColor','flat');
for i = 1:n_sets, bb.CData(i,:) = colors_pastel(mod(i-1,size(colors_pastel,1))+1,:); end
ylabel('Coefficient of Variation (%)'); xlabel('Parameter Set');
set(gca,'FontSize',13);
title('Performance Consistency'); set(gca,'XTick',1:n_sets,'XTickLabel',set_names,'XTickLabelRotation',45);
grid on; box on;
[cvmin, idxmin] = min(cv_pct);
text(idxmin, cvmin+0.15, sprintf('Most\nConsistent\n(%.2f%%)', cvmin), ...
     'HorizontalAlignment','center','Color',[0.6 0 0],'FontWeight','bold');

saveas(figB, sprintf('PubFig_B_EffectSize_Consistency_%s.png', timestamp_str));
savefig(figB, sprintf('PubFig_B_EffectSize_Consistency_%s.fig', timestamp_str));
fprintf('  ✓ PubFig B saved\n');

% ======================================================================
% FIGURE C — Pairwise significance (upper triangle, diagonal hidden)
% ======================================================================
figC = figure('Position',[100 100 900 720],'Color','w');

% compute p-values (two-sample t-test)
p_mat = nan(n_sets);  % NaN diagonal -> blank
for i = 1:n_sets
    for j = i+1:n_sets
        [~,p] = ttest2(all_data_by_set{i}, all_data_by_set{j});
        p_mat(i,j) = p;      % upper triangle only
    end
end

% pastel heatmap map (white -> pastel blue)
t = linspace(0,1,256)'; pastel_map = [1-0.31*t, 1-0.16*t, 1-0.04*t];
imagesc(p_mat,[0 0.05]); colormap(pastel_map);
cb = colorbar; ylabel(cb,'p-value','FontWeight','bold');
axis square; box on;
set(gca,'XTick',1:n_sets,'XTickLabel',set_names,'XTickLabelRotation',45);
set(gca,'YTick',1:n_sets,'YTickLabel',set_names);
set(gca,'LineWidth',0.5, 'XColor',[0.4 0.4 0.4], 'YColor',[0.4 0.4 0.4]);
set(gca,'FontSize',12);
text(n_sets-0.2, 0.7, '* p < 0.05', 'FontSize',11, 'HorizontalAlignment','right', 'Color',[0.2 0.2 0.2]);

title('Pairwise Statistical Significance');

% add stars where significant
hold on;
for i = 1:n_sets
  for j = i+1:n_sets
     if ~isnan(p_mat(i,j)) && p_mat(i,j) < 0.05
        text(j,i,'*','HorizontalAlignment','center','FontSize',14,'FontWeight','bold');
     end
  end
end

saveas(figC, sprintf('PubFig_C_PairwisePvals_%s.png', timestamp_str));
savefig(figC, sprintf('PubFig_C_PairwisePvals_%s.fig', timestamp_str));
fprintf('  ✓ PubFig C saved\n');

% ======================================================================
% FIGURE D — Parameter sensitivity (6 small multiples)  [FIXED]
% ======================================================================
figD = figure('Position',[70 70 1400 1000],'Color','w');

% Build a list of set names robustly (handles cell array or struct array)
if iscell(all_results)
    names_list = cellfun(@(s) s.name, all_results, 'UniformOutput', false);
else
    names_list = {all_results.name};
end

% Helper: find exact name index (case-insensitive fallback; then substring)
findIdx = @(nm) ( ...
    find(strcmp(names_list, nm), 1) ...
    );
if isempty(findIdx('TreesComparison'))
    findIdx = @(nm) ( ...
        find(strcmp(names_list, nm), 1) ...
        );
end
if isempty(findIdx('TreesComparison'))
    findIdx = @(nm) ( ...
        find(strcmpi(names_list, nm), 1) ...
        );
end
if isempty(findIdx('TreesComparison'))
    findIdx = @(nm) ( ...
        find(strcmp(names_list, nm), 1) ...
        );
end

% Convenience to get the i-th results (works if all_results is cell)
getSet = @(ii) ( iscell(all_results) * 0 + 1 ); %#ok<NASGU>
if iscell(all_results)
    getSet = @(ii) all_results{ii};
else
    getSet = @(ii) all_results(ii);
end

all_mean_f1 = mean(cellfun(@mean, all_data_by_set));

% D1: TreesComparison
subplot(3,2,1); hold on;
ii = findIdx('TreesComparison');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    x = [cfg.n_trees]; y = [cfg.test_f1_mean]; e = [cfg.test_f1_std];
    plotcol = colors_pastel(1,:);
    errorbar(x,y,e,'o-','Color',plotcol,'MarkerSize',9,'LineWidth',2.0);
    xlabel('Number of Trees'); ylabel('Test F1 Score'); title('Impact of Number of Trees'); grid on;
    yline(all_mean_f1,'--k','LineWidth',0.8);   % add just below the axis labels block
set(gca,'FontSize',12);
else, axis off; text(0.5,0.5,'No TreesComparison data','HorizontalAlignment','center'); end

% D2: ExtractedFeatures
subplot(3,2,2); hold on;
ii = findIdx('ExtractedFeatures');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    x = [cfg.n_extracted_features]; y = [cfg.test_f1_mean]; e = [cfg.test_f1_std];
    plotcol = colors_pastel(2,:);
    errorbar(x,y,e,'s-','Color',plotcol,'MarkerSize',9,'LineWidth',2.0);
    xlabel('Number of Extracted Features'); ylabel('Test F1 Score'); title('Impact of Extracted Features'); grid on;
yline(all_mean_f1,'--k','LineWidth',0.8);   % add just below the axis labels block
set(gca,'FontSize',12);
else, axis off; text(0.5,0.5,'No ExtractedFeatures data','HorizontalAlignment','center'); end

% D3: SelectedFeatures
subplot(3,2,3); hold on;
ii = findIdx('SelectedFeatures');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    x = [cfg.k_selected_features]; y = [cfg.test_f1_mean]; e = [cfg.test_f1_std];
    plotcol = colors_pastel(3,:);
    errorbar(x,y,e,'^-','Color',plotcol,'MarkerSize',9,'LineWidth',2.0);
    yline(all_mean_f1,'--k','LineWidth',0.8);   % add just below the axis labels block
set(gca,'FontSize',12);
    xlabel('Number of Selected Features'); ylabel('Test F1 Score'); title('Impact of Feature Selection'); grid on;
else, axis off; text(0.5,0.5,'No SelectedFeatures data','HorizontalAlignment','center'); end

% D4: PopulationSizes (μ, λ grid -> pastel heatmap)
subplot(3,2,4); hold on;
ii = findIdx('PopulationSizes');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    mu  = [cfg.mu_numbers]; lam = [cfg.lambda_numbers]; f1 = [cfg.test_f1_mean];
    uMu = unique(mu); uLa = unique(lam);
    F = nan(numel(uMu), numel(uLa));
    for k = 1:numel(cfg)
        F(mu(k)==uMu, lam(k)==uLa) = f1(k);
    end
    imagesc(uLa, uMu, F); set(gca,'YDir','normal');
    % warm pastel colormap (white -> light salmon)
    t = linspace(0,1,256)'; cmapWarm = [1-0.20*t, 1-0.10*t, 1-0.18*t];
    colormap(gca, cmapWarm); colorbar;
    xlabel('\lambda (Offspring)','FontWeight','bold');
ylabel('\mu (Parents)','FontWeight','bold');
set(gca,'FontSize',12); title('ES Population Size Impact');
else, axis off; text(0.5,0.5,'No PopulationSizes data','HorizontalAlignment','center'); end

% D5: OffspringRatio (λ/μ aggregated)
subplot(3,2,5); hold on;
ii = findIdx('OffspringRatio');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    ratio = [cfg.lambda_numbers] ./ max(1,[cfg.mu_numbers]);
    ur = unique(ratio);
    m = zeros(size(ur)); s = zeros(size(ur));
    for r = 1:numel(ur)
        sel = abs(ratio-ur(r)) < 1e-9;
        m(r) = mean([cfg(sel).test_f1_mean]);
        s(r) = std([cfg(sel).test_f1_mean]);
    end
    plotcol = colors_pastel(5,:);
    errorbar(ur, m, s, 'd-','Color',plotcol,'MarkerSize',9,'LineWidth',2.0);
    xlabel('\lambda/\mu Ratio'); ylabel('Test F1 Score');yline(all_mean_f1,'--k','LineWidth',0.8);   % add just below the axis labels block
set(gca,'FontSize',12); title('Impact of Offspring Ratio'); grid on;
else, axis off; text(0.5,0.5,'No OffspringRatio data','HorizontalAlignment','center'); end

% D6: TestRatioComparison
subplot(3,2,6); hold on;
ii = findIdx('TestRatioComparison');
if ~isempty(ii)
    S = getSet(ii); cfg = S.configs;
    x = [cfg.test_ratio]; y = [cfg.test_f1_mean]; e = [cfg.test_f1_std];
    plotcol = colors_pastel(6,:);
    errorbar(x,y,e,'v-','Color',plotcol,'MarkerSize',9,'LineWidth',2.0);yline(all_mean_f1,'--k','LineWidth',0.8);   % add just below the axis labels block
set(gca,'FontSize',12);
    xlabel('Test Ratio'); ylabel('Test F1 Score'); title('Impact of Train/Test Split'); grid on;
else, axis off; text(0.5,0.5,'No TestRatioComparison data','HorizontalAlignment','center'); end

saveas(figD, sprintf('PubFig_D_Sensitivity_%s.png', timestamp_str));
savefig(figD, sprintf('PubFig_D_Sensitivity_%s.fig', timestamp_str));
fprintf('  ✓ PubFig D saved\n');

% ======================================================================
% FIGURE E (Optional) — Best config per set (points+jitter) + Overall mean (barh+CI)
% ======================================================================
figE = figure('Position',[90 90 1400 600],'Color','w');

% E1: Best config runs (jittered points) + mean±std
subplot(1,2,1); hold on;
for i = 1:n_sets
    runs = all_results{i}.best_config.all_runs.test_f1;
    xj = i + 0.20*(rand(size(runs))-0.5);
    col = colors_pastel(mod(i-1,size(colors_pastel,1))+1,:);
    scatter(xj, runs, 36, col,'filled','MarkerFaceAlpha',0.6);
    errorbar(i, mean(runs), std(runs), 'ko','MarkerFaceColor','k','LineWidth',1.4,'MarkerSize',7);
end
xlim([0.5 n_sets+0.5]); ylim([0.82 0.92]); grid on; box on;
set(gca,'XTick',1:n_sets,'XTickLabel',set_names,'XTickLabelRotation',45);
ylabel('Test F1 Score'); xlabel('Parameter Set');
set(gca,'FontSize',13);
title('Best Configuration per Parameter Set (All Runs)');

% E2: Mean across ALL configs with 95% CI (barh)
subplot(1,2,2); hold on;
mean_all = zeros(n_sets,1); std_all = zeros(n_sets,1); n_all = zeros(n_sets,1);
for i = 1:n_sets
    vals = all_data_by_set{i};
    mean_all(i) = mean(vals); std_all(i) = std(vals); n_all(i) = numel(vals);
end
ci95 = 1.96 * std_all ./ sqrt(max(1,n_all));
[~,ord] = sort(mean_all,'descend');
bh = barh(1:n_sets, mean_all(ord), 'FaceColor','flat');
for i = 1:n_sets
    bh.CData(i,:) = colors_pastel(mod(ord(i)-1,size(colors_pastel,1))+1,:);
end
errorbar(mean_all(ord), 1:n_sets, ci95(ord), 'horizontal','k','LineStyle','none','LineWidth',1.4);
set(gca,'YTick',1:n_sets,'YTickLabel',set_names(ord));
xline(mean(mean_all),'--k','LineWidth',1);   % global mean
set(gca,'FontSize',13);
title('Overall Performance with 95% CI','FontSize',14);
xlabel('Mean Test F1 Score Across All Configurations'); ylabel('Parameter Set');
xlim([min(mean_all-ci95)-0.01, max(mean_all+ci95)+0.01]);
saveas(figE, sprintf('PubFig_E_BestAndMeans_%s.png', timestamp_str));
savefig(figE, sprintf('PubFig_E_BestAndMeans_%s.fig', timestamp_str));
fprintf('  ✓ PubFig E saved (optional)\n');

fprintf('=== DONE: New publication figures generated ===\n');


%% === CREATE & SAVE PUBLICATION TABLES (ROBUST) =========================
outdir = 'tables';
if ~exist(outdir,'dir'), mkdir(outdir); end

nSets = numel(all_results);

names      = strings(nSets,1);
testF1_m   = zeros(nSets,1);
testF1_s   = zeros(nSets,1);
testAcc_m  = zeros(nSets,1);
testAcc_s  = zeros(nSets,1);
trainF1_m  = zeros(nSets,1);
trainF1_s  = zeros(nSets,1);
cv_pct     = zeros(nSets,1);
nConfigs   = zeros(nSets,1);
nRuns      = zeros(nSets,1);

% best hyperparams per set
T_best  = zeros(nSets,1);
E_best  = zeros(nSets,1);
S_best  = zeros(nSets,1);
MU_best = zeros(nSets,1);
LA_best = zeros(nSets,1);
TR_best = zeros(nSets,1);

for i = 1:nSets
    S = all_results{i};
    B = S.best_config;

    names(i)     = string(strrep(S.name,'_',' '));
    testF1_m(i)  = B.test_f1_mean;
    testF1_s(i)  = B.test_f1_std;
    testAcc_m(i) = B.test_accuracy_mean;
    testAcc_s(i) = B.test_accuracy_std;
    trainF1_m(i) = B.train_f1_mean;
    trainF1_s(i) = B.train_f1_std;

    % CV% (guard against divide-by-zero)
    cv_pct(i)    = 100 * (B.test_f1_std / max(eps, B.test_f1_mean));

    nConfigs(i)  = numel(S.configs);
    nRuns(i)     = getRunsSafe(B, S);   % <<< robust run counter

    % hyperparams (present for RF+ES)
    if isfield(B,'n_trees'),                T_best(i)  = B.n_trees; else,  T_best(i)  = NaN; end
    if isfield(B,'n_extracted_features'),   E_best(i)  = B.n_extracted_features; else, E_best(i) = NaN; end
    if isfield(B,'k_selected_features'),    S_best(i)  = B.k_selected_features; else, S_best(i) = NaN; end
    if isfield(B,'mu_numbers'),             MU_best(i) = B.mu_numbers; else, MU_best(i) = NaN; end
    if isfield(B,'lambda_numbers'),         LA_best(i) = B.lambda_numbers; else, LA_best(i) = NaN; end
    if isfield(B,'test_ratio'),             TR_best(i) = B.test_ratio; else, TR_best(i) = NaN; end
end

% ---------- Table A: best configuration performance per set ----------
TblBest = table( ...
    names, ...
    testF1_m, testF1_s, ...
    testAcc_m, testAcc_s, ...
    trainF1_m, trainF1_s, ...
    cv_pct, nConfigs, nRuns, ...
    'VariableNames', ["ParameterSet","TestF1_Mean","TestF1_Std", ...
                      "TestAcc_Mean","TestAcc_Std", ...
                      "TrainF1_Mean","TrainF1_Std", ...
                      "CV_percent","Configs","Runs"]);

% ---------- Table B: ranking by Test F1 ----------
[~,ord] = sort(testF1_m,'descend');
TblRank = table( (1:nSets).', names(ord), testF1_m(ord), cv_pct(ord), ...
    'VariableNames', ["Rank","ParameterSet","TestF1_Mean","CV_percent"]);

% ---------- Table C: best hyperparameters per set ----------
TblHyper = table( ...
    names, T_best, E_best, S_best, MU_best, LA_best, TR_best, ...
    'VariableNames', ["ParameterSet","Trees","ExtractedFeat","SelectedFeat","Mu","Lambda","TestRatio"]);

% ---------- Table D: experiment scale ----------
TblScale = table(sum(nConfigs), total_runs_all, ...
    'VariableNames', ["TotalConfigurations","TotalRuns"]);

% ================= SAVE CSV / EXCEL =================
writetable(TblBest,  fullfile(outdir, sprintf('BestConfigSummary_%s.csv', timestamp_str)));
writetable(TblRank,  fullfile(outdir, sprintf('Ranking_%s.csv',           timestamp_str)));
writetable(TblHyper, fullfile(outdir, sprintf('BestHyperparams_%s.csv',   timestamp_str)));
writetable(TblScale, fullfile(outdir, sprintf('ExperimentScale_%s.csv',   timestamp_str)));

xlsx = fullfile(outdir, sprintf('ResultsTables_%s.xlsx', timestamp_str));
writetable(TblBest,  xlsx, 'Sheet','BestSummary');
writetable(TblRank,  xlsx, 'Sheet','Ranking','WriteMode','overwrite');
writetable(TblHyper, xlsx, 'Sheet','BestHyperparams','WriteMode','overwrite');
writetable(TblScale, xlsx, 'Sheet','ExperimentScale','WriteMode','overwrite');

fprintf('  ✓ Tables saved to %s (CSV + Excel)\n', outdir);

% ================= (OPTIONAL) SAVE LaTeX =================
doLatex = true;  % set false if you don't need .tex
if doLatex
    texA = fullfile(outdir, sprintf('BestConfigSummary_%s.tex', timestamp_str));
    texB = fullfile(outdir, sprintf('Ranking_%s.tex',           timestamp_str));
    texC = fullfile(outdir, sprintf('BestHyperparams_%s.tex',   timestamp_str));

    latex_write_best(texA, TblBest, ...
      'Parameter Optimization Results — Best Configuration per Set', 'tab:best_summary');
    latex_write_rank(texB, TblRank, ...
      'Ranking by Test F1 (best configuration per set)', 'tab:ranking');
    latex_write_hyper(texC, TblHyper, ...
      'Winning hyperparameters per parameter set', 'tab:best_hyper');

    fprintf('  ✓ LaTeX tables written to %s\n', outdir);
end

%% ----------------- LOCAL HELPERS ---------------------------------------
function n = getRunsSafe(cfg, S)
% Robustly determine number of runs for a best_config `cfg`.
% `S` (whole set) is optional and used for fallbacks.
    if isstruct(cfg) && isfield(cfg,'total_runs_completed')
        n = cfg.total_runs_completed;
        return;
    end
    if isfield(cfg,'all_runs')
        if isfield(cfg.all_runs,'test_f1') && ~isempty(cfg.all_runs.test_f1)
            n = numel(cfg.all_runs.test_f1); return;
        elseif isfield(cfg.all_runs,'test_accuracy') && ~isempty(cfg.all_runs.test_accuracy)
            n = numel(cfg.all_runs.test_accuracy); return;
        end
    end
    if nargin>=2 && isfield(S,'configs') && ~isempty(S.configs) ...
            && isfield(S.configs(1),'all_runs') && isfield(S.configs(1).all_runs,'test_f1')
        n = numel(S.configs(1).all_runs.test_f1); return;
    end
    n = NaN; % last resort
end

function latex_write_best(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires: \\usepackage{booktabs}\\usepackage{siunitx}\n');
fprintf(fid, '\\begin{table}[ht]\\centering\n');
fprintf(fid, '\\caption{%s}\\label{%s}\n', caption, label);
fprintf(fid, ['\\begin{tabular}{l' ...
              ' S[table-format=1.3(3)]' ...
              ' S[table-format=1.3(3)]' ...
              ' S[table-format=1.3(3)]' ...
              ' S[table-format=2.2]' ...
              ' S[table-format=2.0]' ...
              ' S[table-format=2.0]}\n']);
fprintf(fid, '\\toprule\n');
fprintf(fid, ['\\textbf{Parameter Set} & {\\textbf{Test F1}} & {\\textbf{Test Acc}} & ' ...
              '{\\textbf{Train F1}} & {\\textbf{CV\\%%}} & {\\textbf{Configs}} & {\\textbf{Runs}} \\\\\n']);
fprintf(fid, '\\midrule\n');
for i = 1:height(T)
    nm = T.ParameterSet(i);
    f1 = sprintf('%.3f \\pm %.3f', T.TestF1_Mean(i),  T.TestF1_Std(i));
    ac = sprintf('%.3f \\pm %.3f', T.TestAcc_Mean(i), T.TestAcc_Std(i));
    tr = sprintf('%.3f \\pm %.3f', T.TrainF1_Mean(i), T.TrainF1_Std(i));
    fprintf(fid, '%s & %s & %s & %s & %.2f & %d & %d \\\\\n', ...
        nm, f1, ac, tr, T.CV_percent(i), T.Configs(i), T.Runs(i));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
end

function latex_write_rank(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires: \\usepackage{booktabs}\\usepackage{siunitx}\n');
fprintf(fid, '\\begin{table}[ht]\\centering\n');
fprintf(fid, '\\caption{%s}\\label{%s}\n', caption, label);
fprintf(fid, '\\begin{tabular}{c l S[table-format=1.3] S[table-format=2.2]}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, '\\textbf{Rank} & \\textbf{Parameter Set} & {\\textbf{Test F1}} & {\\textbf{CV\\%%}} \\\\\n');
fprintf(fid, '\\midrule\n');
for i = 1:height(T)
   fprintf(fid, '%d & %s & %.3f & %.2f \\\\\n', T.Rank(i), T.ParameterSet(i), T.TestF1_Mean(i), T.CV_percent(i));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
end

function latex_write_hyper(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires: \\usepackage{booktabs}\\usepackage{siunitx}\n');
fprintf(fid, '\\begin{table}[ht]\\centering\n');
fprintf(fid, '\\caption{%s}\\label{%s}\n', caption, label);
fprintf(fid, '\\begin{tabular}{l S[table-format=3.0] S[table-format=2.0] S[table-format=2.0] S[table-format=3.0] S[table-format=3.0] S[table-format=1.2]}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, ['\\textbf{Parameter Set} & \\textbf{Trees} & \\textbf{Extr.} & \\textbf{Sel.} & ' ...
              '\\boldmath{$\\mu$} & \\boldmath{$\\lambda$} & \\textbf{TestRatio} \\\\\n']);
fprintf(fid, '\\midrule\n');
for i = 1:height(T)
    fprintf(fid, '%s & %d & %d & %d & %d & %d & %.2f \\\\\n', ...
        T.ParameterSet(i), T.Trees(i), T.ExtractedFeat(i), T.SelectedFeat(i), ...
        T.Mu(i), T.Lambda(i), T.TestRatio(i));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
end
function total = count_total_runs(all_results)
% Robustly sum runs over every configuration in every set.
% Works whether results carry total_runs_completed or only all_runs.* fields.

    total = 0;

    for i = 1:numel(all_results)
        S = all_results{i};
        for c = 1:numel(S.configs)
            cfg = S.configs(c);

            if isfield(cfg,'total_runs_completed') && ~isempty(cfg.total_runs_completed)
                total = total + double(cfg.total_runs_completed);

            elseif isfield(cfg,'all_runs')
                if isfield(cfg.all_runs,'test_f1') && ~isempty(cfg.all_runs.test_f1)
                    total = total + numel(cfg.all_runs.test_f1);
                elseif isfield(cfg.all_runs,'test_accuracy') && ~isempty(cfg.all_runs.test_accuracy)
                    total = total + numel(cfg.all_runs.test_accuracy);
                end
            end
        end
    end

    % Fallback — if still zero/empty, assume constant runs-per-config within a set
    if isempty(total) || ~isfinite(total) || total==0
        total = 0;
        for i = 1:numel(all_results)
            S = all_results{i};
            if isfield(S,'configs') && ~isempty(S.configs)
                runs_per_cfg = NaN;

                % try to infer runs per config from first config
                cfg1 = S.configs(1);
                if isfield(cfg1,'total_runs_completed') && ~isempty(cfg1.total_runs_completed)
                    runs_per_cfg = double(cfg1.total_runs_completed);
                elseif isfield(cfg1,'all_runs')
                    if isfield(cfg1.all_runs,'test_f1') && ~isempty(cfg1.all_runs.test_f1)
                        runs_per_cfg = numel(cfg1.all_runs.test_f1);
                    elseif isfield(cfg1.all_runs,'test_accuracy') && ~isempty(cfg1.all_runs.test_accuracy)
                        runs_per_cfg = numel(cfg1.all_runs.test_accuracy);
                    end
                end

                if ~isnan(runs_per_cfg)
                    total = total + numel(S.configs) * runs_per_cfg;
                end
            end
        end
    end
end
