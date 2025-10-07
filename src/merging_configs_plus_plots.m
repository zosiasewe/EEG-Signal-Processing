clc
clear
close all

%% ===============================================================
%  ANALYSIS - Load merged results, summarize, plot, and export
%  ===============================================================
timestamp_str = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
fprintf('=== Results Analysis & Plotting ===\n');
fprintf('Timestamp: %s\n\n', timestamp_str);

% Hide axes toolbar in exported images
try, set(groot,'defaultAxesToolbarVisible','off'); end %#ok<TRYNC>

% ---------- Which sets to load ----------
all_set_names = {'PopulationSizes','OffspringRatio','TestRatioComparison',...
                 'TreesComparison','ExtractedFeatures','SelectedFeatures'};

all_results = {};

%% ---------------------------------------------------------------
% 1) Load ALL results (already merged or complete)
% ---------------------------------------------------------------
fprintf('Loading results files...\n');
for i = 1:numel(all_set_names)
    set_name = all_set_names{i};
    
    % First try to find MERGED file
    merged_files = dir(sprintf('results_%s_MERGED_*.mat', set_name));
    
    if ~isempty(merged_files)
        % Use the most recent merged file
        [~, idx] = max([merged_files.datenum]);
        data = load(merged_files(idx).name);
        S = data.set_results;
        if ~isfield(S,'name') || isempty(S.name), S.name = set_name; end
        all_results{end+1} = S; %#ok<SAGROW>
        
        if isfield(S.configs(1),'total_runs_completed')
            n_runs = S.configs(1).total_runs_completed;
        else
            n_runs = numel(S.configs(1).all_runs.test_f1);
        end
        fprintf('  ✓ %s: %d configs, %d runs each (MERGED file: %s)\n', ...
            set_name, numel(S.configs), n_runs, merged_files(idx).name);
    else
        % Try regular results file (non-batch, non-merged)
        files = dir(sprintf('results_%s_*.mat', set_name));
        % Exclude batch files and MERGED files
        files = files(~contains({files.name}, 'batch') & ~contains({files.name}, 'MERGED'));
        
        if ~isempty(files)
            % Use the most recent file
            [~, idx] = max([files.datenum]);
            data = load(files(idx).name);
            S = data.set_results;
            if ~isfield(S,'name') || isempty(S.name), S.name = set_name; end
            all_results{end+1} = S; %#ok<SAGROW>
            
            if isfield(S.configs(1),'total_runs_completed')
                n_runs = S.configs(1).total_runs_completed;
            else
                n_runs = numel(S.configs(1).all_runs.test_f1);
            end
            fprintf('  ✓ %s: %d configs, %d runs each (regular file: %s)\n', ...
                set_name, numel(S.configs), n_runs, files(idx).name);
        else
            fprintf('  ✗ WARNING: No file found for %s\n', set_name);
        end
    end
end

if isempty(all_results)
    error('No results files found! Please check your directory.');
end

%% ---------------------------------------------------------------
% 2) Verification + standardize
% ---------------------------------------------------------------
fprintf('\n=== VERIFICATION ===\n');
total_configs = 0;
for i = 1:numel(all_results)
    S = all_results{i};
    n_configs = numel(S.configs);
    if isfield(S.configs(1),'total_runs_completed')
        n_runs = S.configs(1).total_runs_completed;
    else
        n_runs = numel(S.configs(1).all_runs.test_f1);
    end
    total_configs = total_configs + n_configs;
    fprintf('%s: %d configs × %d runs = %d total runs\n', ...
        S.name, n_configs, n_runs, n_configs*n_runs);
end
fprintf('TOTAL: %d configurations\n', total_configs);

fprintf('\nStandardizing result structures...\n');
for i = 1:numel(all_results)
    for c = 1:numel(all_results{i}.configs)
        cfg = all_results{i}.configs(c);
        if ~isfield(cfg,'total_runs_completed') || isempty(cfg.total_runs_completed)
            all_results{i}.configs(c).total_runs_completed = numel(cfg.all_runs.test_f1);
        end
        % ensure polygon means exist even on older files
        if ~isfield(cfg,'test_polygon_area_mean') || isempty(cfg.test_polygon_area_mean)
            all_results{i}.configs(c).test_polygon_area_mean = safe_mean(cfg.all_runs,'test_polygon_area');
        end
        if ~isfield(cfg,'test_polygon_area_std') || isempty(cfg.test_polygon_area_std)
            all_results{i}.configs(c).test_polygon_area_std = safe_std(cfg.all_runs,'test_polygon_area');
        end
    end
end
fprintf('  ✓ All results standardized\n');



% ----- Console summary -----
fprintf('Best Configuration Found:\n');
fprintf('  Parameter Set: %s\n', overall_best_set);

if isfield(overall_best_config,'test_f1_mean') && isfield(overall_best_config,'test_f1_std')
    fprintf('  Test F1: %.4f ± %.4f\n', ...
        overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
end

if isfield(overall_best_config,'test_accuracy_mean') && isfield(overall_best_config,'test_accuracy_std')
    fprintf('  Test Accuracy: %.4f ± %.4f\n', ...
        overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
end

if isfield(overall_best_config,'train_f1_mean') && isfield(overall_best_config,'train_f1_std')
    fprintf('  Train F1: %.4f ± %.4f\n', ...
        overall_best_config.train_f1_mean, overall_best_config.train_f1_std);
end

% Polygon metric if available
if isfield(overall_best_config,'test_polygon_area_mean') && isfield(overall_best_config,'test_polygon_area_std')
    fprintf('  Polygon Area: %.4f ± %.4f\n', ...
        overall_best_config.test_polygon_area_mean, overall_best_config.test_polygon_area_std);
end

fprintf('  Configuration:\n');
if isfield(overall_best_config,'n_trees')
    fprintf('    Trees: %d\n', overall_best_config.n_trees);
end
if isfield(overall_best_config,'n_extracted_features')
    fprintf('    Extracted Features: %d\n', overall_best_config.n_extracted_features);
end
if isfield(overall_best_config,'k_selected_features')
    fprintf('    Selected Features: %d\n', overall_best_config.k_selected_features);
end
if isfield(overall_best_config,'mu_numbers')
    fprintf('    μ (parents): %d\n', overall_best_config.mu_numbers);
end
if isfield(overall_best_config,'lambda_numbers')
    fprintf('    λ (offspring): %d\n', overall_best_config.lambda_numbers);
end
if isfield(overall_best_config,'test_ratio')
    fprintf('    Test Ratio: %.2f\n', overall_best_config.test_ratio);
end

%% ---------------------------------------------------------------
% 5) Compute robust TOTAL RUNS (no double count)
% ---------------------------------------------------------------
total_runs_all = count_total_runs(all_results);

%% ---------------------------------------------------------------
% 6) Generate TEXT SUMMARY (with polygon)
% ---------------------------------------------------------------
fprintf('n=== CREATING TEXT SUMMARY ===n');
fid = fopen(sprintf('Results_Summary_%s.txt', timestamp_str), 'w');

fprintf(fid, '============================================n');
fprintf(fid, 'COMPREHENSIVE PARAMETER OPTIMIZATION RESULTSn');
fprintf(fid, '============================================n');
fprintf(fid, 'Generated %snn', timestamp_str);

fprintf(fid, 'OVERALL BEST CONFIGURATIONn');
fprintf(fid, '  Parameter Set %sn', overall_best_set);
fprintf(fid, '  Test F1 Score %.4f ± %.4fn', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf(fid, '  Test Accuracy %.4f ± %.4fn', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf(fid, '  Train F1 Score %.4f ± %.4fn', overall_best_config.train_f1_mean, overall_best_config.train_f1_std);
[pa_m, pa_s] = best_polygon(overall_best_config);
fprintf(fid, '  Polygon Area %.4f ± %.4fnn', pa_m, pa_s);

fprintf(fid, '  Hyperparametersn');
fprintf(fid, '    - Number of Trees %dn', overall_best_config.n_trees);
fprintf(fid, '    - Extracted Features %dn', overall_best_config.n_extracted_features);
fprintf(fid, '    - Selected Features %dn', overall_best_config.k_selected_features);
fprintf(fid, '    - ES Parents (μ) %dn', overall_best_config.mu_numbers);
fprintf(fid, '    - ES Offspring (λ) %dn', overall_best_config.lambda_numbers);
fprintf(fid, '    - Test Ratio %.2fnn', overall_best_config.test_ratio);

fprintf(fid, '============================================n');
fprintf(fid, 'RESULTS BY PARAMETER SETn');
fprintf(fid, '============================================nn');

for i = 1numel(all_results)
    S = all_results{i};
    fprintf(fid, '%d. %sn', i, S.name);
    fprintf(fid, '   Configurations tested %dn', numel(S.configs));
    fprintf(fid, '   Runs per configuration %dn', S.configs(1).total_runs_completed);

    B = S.best_config;
    [bp_m,bp_s] = best_polygon(B);

    fprintf(fid, 'n   Best Configurationn');
    fprintf(fid, '     Test F1 %.4f ± %.4fn', B.test_f1_mean, B.test_f1_std);
    fprintf(fid, '     Test Acc %.4f ± %.4fn', B.test_accuracy_mean, B.test_accuracy_std);
    fprintf(fid, '     Polygon Area %.4f ± %.4fn', bp_m, bp_s);
    fprintf(fid, '     Parameters T=%d, E=%d, S=%d, μ=%d, λ=%d, TestRatio=%.2fn', ...
        B.n_trees, B.n_extracted_features, B.k_selected_features, ...
        B.mu_numbers, B.lambda_numbers, B.test_ratio);

    fprintf(fid, 'n   All Configurations (F1 Mean ± Std)n');
    for c = 1numel(S.configs)
        cfg = S.configs(c);
        fprintf(fid, '     Config %d %.4f ± %.4f  [T=%d, E=%d, S=%d, μ=%d, λ=%d, TR=%.2f]n', ...
            c, cfg.test_f1_mean, cfg.test_f1_std, ...
            cfg.n_trees, cfg.n_extracted_features, cfg.k_selected_features, ...
            cfg.mu_numbers, cfg.lambda_numbers, cfg.test_ratio);
    end
    fprintf(fid, 'n');
end

fprintf(fid, '============================================n');
fprintf(fid, 'STATISTICAL SUMMARYn');
fprintf(fid, '============================================nn');

fprintf(fid, 'Total Configurations %dn', total_configs);
fprintf(fid, 'Total Experimental Runs %dn', total_runs_all);

% global F1 stats (per-config means)
all_f1_means = [];
for i = 1numel(all_results)
    all_f1_means = [all_f1_means; [all_results{i}.configs.test_f1_mean]']; %#okAGROW
end
fprintf(fid, 'nF1 Score Statistics Across All Configurationsn');
fprintf(fid, '  Mean %.4fn', mean(all_f1_means));
fprintf(fid, '  Std %.4fn',  std(all_f1_means));
fprintf(fid, '  Min %.4fn',  min(all_f1_means));
fprintf(fid, '  Max %.4fn',  max(all_f1_means));
fprintf(fid, '  Median %.4fn', median(all_f1_means));

% global Polygon Area stats (per-config means, fallback to runs if needed)
all_poly_means = [];
for i = 1numel(all_results)
    if isfield(all_results{i}.configs,'test_polygon_area_mean')
        all_poly_means = [all_poly_means; [all_results{i}.configs.test_polygon_area_mean]']; %#okAGROW
    else
        cfgs = all_results{i}.configs;
        tmp = NaN(numel(cfgs),1);
        for c = 1numel(cfgs)
            tmp(c) = safe_mean(cfgs(c).all_runs,'test_polygon_area');
        end
        all_poly_means = [all_poly_means; tmp(~isnan(tmp))]; %#okAGROW
    end
end
fprintf(fid, 'nPolygon Area Statistics Across All Configurationsn');
fprintf(fid, '  Mean %.4fn', mean(all_poly_means));
fprintf(fid, '  Std %.4fn',  std(all_poly_means));
fprintf(fid, '  Min %.4fn',  min(all_poly_means));
fprintf(fid, '  Max %.4fn',  max(all_poly_means));
fprintf(fid, '  Median %.4fn', median(all_poly_means));

fclose(fid);

%% 7) CREATE COMPREHENSIVE TEXT SUMMARY (robust + polygon metric)
fprintf('n=== CREATING TEXT SUMMARY ===n');

summary_path = sprintf('Results_Summary_%s.txt', timestamp_str);
fid = fopen(summary_path, 'w');

fprintf(fid, '============================================n');
fprintf(fid, 'COMPREHENSIVE PARAMETER OPTIMIZATION RESULTSn');
fprintf(fid, '============================================n');
fprintf(fid, 'Generated %snn', timestamp_str);

% ----- OVERALL BEST CONFIG -----
fprintf(fid, 'OVERALL BEST CONFIGURATIONn');
fprintf(fid, '  Parameter Set %sn', overall_best_set);

if isfield(overall_best_config,'test_f1_mean') && isfield(overall_best_config,'test_f1_std')
    fprintf(fid, '  Test F1 Score %.4f ± %.4fn', ...
        overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
else
    fprintf(fid, '  Test F1 Score --n');
end
if isfield(overall_best_config,'test_accuracy_mean') && isfield(overall_best_config,'test_accuracy_std')
    fprintf(fid, '  Test Accuracy %.4f ± %.4fn', ...
        overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
else
    fprintf(fid, '  Test Accuracy --n');
end
if isfield(overall_best_config,'train_f1_mean') && isfield(overall_best_config,'train_f1_std')
    fprintf(fid, '  Train F1 Score %.4f ± %.4fn', ...
        overall_best_config.train_f1_mean, overall_best_config.train_f1_std);
else
    fprintf(fid, '  Train F1 Score --n');
end
if isfield(overall_best_config,'test_polygon_area_mean') && isfield(overall_best_config,'test_polygon_area_std') ...
        && ~isempty(overall_best_config.test_polygon_area_mean)
    fprintf(fid, '  Polygon Area %.4f ± %.4fn', ...
        overall_best_config.test_polygon_area_mean, overall_best_config.test_polygon_area_std);
else
    fprintf(fid, '  Polygon Area --n');
end
fprintf(fid, 'n  Hyperparametersn');
if isfield(overall_best_config,'n_trees'),              fprintf(fid, '    - Number of Trees %dn', overall_best_config.n_trees); end
if isfield(overall_best_config,'n_extracted_features'), fprintf(fid, '    - Extracted Features %dn', overall_best_config.n_extracted_features); end
if isfield(overall_best_config,'k_selected_features'),  fprintf(fid, '    - Selected Features %dn', overall_best_config.k_selected_features); end
if isfield(overall_best_config,'mu_numbers'),           fprintf(fid, '    - ES Parents (μ) %dn', overall_best_config.mu_numbers); end
if isfield(overall_best_config,'lambda_numbers'),       fprintf(fid, '    - ES Offspring (λ) %dn', overall_best_config.lambda_numbers); end
if isfield(overall_best_config,'test_ratio'),           fprintf(fid, '    - Test Ratio %.2fn', overall_best_config.test_ratio); end
fprintf(fid, 'n');

% ----- RESULTS BY PARAMETER SET -----
fprintf(fid, '============================================n');
fprintf(fid, 'RESULTS BY PARAMETER SETn');
fprintf(fid, '============================================nn');

for i = 1length(all_results)
    S = all_results{i};
    fprintf(fid, '%d. %sn', i, S.name);

    n_cfg = numel(S.configs);
    runs_per_cfg = getRunsSafe(S.configs(1), S);
    if isnan(runs_per_cfg), runs_per_cfg = 0; end

    fprintf(fid, '   Configurations tested %dn', n_cfg);
    fprintf(fid, '   Runs per configuration %dn', runs_per_cfg);

    % Best config for this set
    B = S.best_config;
    fprintf(fid, 'n   Best Configurationn');
    if isfield(B,'test_f1_mean') && isfield(B,'test_f1_std')
        fprintf(fid, '     Test F1 %.4f ± %.4fn', B.test_f1_mean, B.test_f1_std);
    else
        fprintf(fid, '     Test F1 --n');
    end
    if isfield(B,'test_accuracy_mean') && isfield(B,'test_accuracy_std')
        fprintf(fid, '     Test Acc %.4f ± %.4fn', B.test_accuracy_mean, B.test_accuracy_std);
    else
        fprintf(fid, '     Test Acc --n');
    end
    if isfield(B,'test_polygon_area_mean') && isfield(B,'test_polygon_area_std') ...
            && ~isempty(B.test_polygon_area_mean)
        fprintf(fid, '     Polygon Area %.4f ± %.4fn', B.test_polygon_area_mean, B.test_polygon_area_std);
    else
        fprintf(fid, '     Polygon Area --n');
    end

    % Hyperparameters (print only when present)
    fprintf(fid, '     Parameters');
    if isfield(B,'n_trees'),                fprintf(fid, ' T=%d,',   B.n_trees); end
    if isfield(B,'n_extracted_features'),   fprintf(fid, ' E=%d,',   B.n_extracted_features); end
    if isfield(B,'k_selected_features'),    fprintf(fid, ' S=%d,',   B.k_selected_features); end
    if isfield(B,'mu_numbers'),             fprintf(fid, ' μ=%d,',   B.mu_numbers); end
    if isfield(B,'lambda_numbers'),         fprintf(fid, ' λ=%d,',   B.lambda_numbers); end
    if isfield(B,'test_ratio'),             fprintf(fid, ' TestRatio=%.2f,', B.test_ratio); end
    % trim trailing comma
    fprintf(fid, 'b n');

    % All configs in the set
    fprintf(fid, 'n   All Configurations (F1 Mean ± Std; Polygon Area)n');
    for c = 1n_cfg
        C = S.configs(c);

        % F1 string
        if isfield(C,'test_f1_mean') && isfield(C,'test_f1_std')
            f1_str = sprintf('%.4f ± %.4f', C.test_f1_mean, C.test_f1_std);
        else
            f1_str = '--';
        end

        % Polygon string (optional)
        if isfield(C,'test_polygon_area_mean') && isfield(C,'test_polygon_area_std') ...
                && ~isempty(C.test_polygon_area_mean)
            poly_str = sprintf('PA=%.4f ± %.4f', C.test_polygon_area_mean, C.test_polygon_area_std);
        else
            poly_str = 'PA=--';
        end

        % Hyperparams compact
        hp = '';
        if isfield(C,'n_trees'),              hp = [hp, sprintf('T=%d, ',   C.n_trees)]; end
        if isfield(C,'n_extracted_features'), hp = [hp, sprintf('E=%d, ',   C.n_extracted_features)]; end
        if isfield(C,'k_selected_features'),  hp = [hp, sprintf('S=%d, ',   C.k_selected_features)]; end
        if isfield(C,'mu_numbers'),           hp = [hp, sprintf('μ=%d, ',   C.mu_numbers)]; end
        if isfield(C,'lambda_numbers'),       hp = [hp, sprintf('λ=%d, ',   C.lambda_numbers)]; end
        if isfield(C,'test_ratio'),           hp = [hp, sprintf('TR=%.2f, ',C.test_ratio)]; end
        if ~isempty(hp), hp = hp(1:end-2); end  % strip trailing comma+space

        fprintf(fid, '     Config %d %s  [%s; %s]n', c, f1_str, hp, poly_str);
    end
    fprintf(fid, 'n');
end

% ----- GLOBAL STATS -----
fprintf(fid, '============================================n');
fprintf(fid, 'STATISTICAL SUMMARYn');
fprintf(fid, '============================================nn');

fprintf(fid, 'Total Configurations %dn', total_configs);
fprintf(fid, 'Total Experimental Runs %dn', total_runs_all);

% F1 stats across all configs
all_f1_means = [];
for i = 1length(all_results)
    if isfield(all_results{i}.configs, 'test_f1_mean')
        all_f1_means = [all_f1_means; [all_results{i}.configs.test_f1_mean]']; %#okAGROW
    end
end
if ~isempty(all_f1_means)
    fprintf(fid, 'nF1 Score Statistics Across All Configurationsn');
    fprintf(fid, '  Mean %.4fn', mean(all_f1_means));
    fprintf(fid, '  Std %.4fn',  std(all_f1_means));
    fprintf(fid, '  Min %.4fn',  min(all_f1_means));
    fprintf(fid, '  Max %.4fn',  max(all_f1_means));
    fprintf(fid, '  Median %.4fn', median(all_f1_means));
end

% Polygon stats across all configs (only if present)
all_poly_means = [];
for i = 1length(all_results)
    Ci = all_results{i}.configs;
    if isfield(Ci, 'test_polygon_area_mean')
        vec = [Ci.test_polygon_area_mean]';
        vec = vec(~isnan(vec));
        all_poly_means = [all_poly_means; vec]; %#okAGROW
    end
end
if ~isempty(all_poly_means)
    fprintf(fid, 'nPolygon Area Statistics Across All Configurationsn');
    fprintf(fid, '  Mean %.4fn', mean(all_poly_means));
    fprintf(fid, '  Std %.4fn',  std(all_poly_means));
    fprintf(fid, '  Min %.4fn',  min(all_poly_means));
    fprintf(fid, '  Max %.4fn',  max(all_poly_means));
    fprintf(fid, '  Median %.4fn', median(all_poly_means));
end

fclose(fid);

%% 8. CREATE LATEX TABLE FOR PUBLICATION (robust + polygon metric)
fprintf('n=== CREATING LATEX TABLE ===n');

fid_tex = fopen(sprintf('Results_Table_%s.tex', timestamp_str), 'w');

fprintf(fid_tex, 'begin{table}[htbp]n');
fprintf(fid_tex, 'centeringn');
fprintf(fid_tex, 'caption{Parameter Optimization Results -- Best Configuration per Parameter Set}n');
fprintf(fid_tex, 'label{tabresults}n');
fprintf(fid_tex, 'begin{tabular}{lccccccc}n');
fprintf(fid_tex, 'hlinen');
fprintf(fid_tex, 'textbf{Parameter Set} & textbf{Test F1} & textbf{Test Acc} & textbf{Train F1} & textbf{Polygon Area} & textbf{Configs} & textbf{Runs} n');
fprintf(fid_tex, 'hlinen');

for i = 1length(all_results)
    B = all_results{i}.best_config;                   % - local best config
    set_name_clean = strrep(all_results{i}.name, '_', '_');

    % robust runs
    n_runs = getRunsSafe(B, all_results{i});
    if isnan(n_runs), n_runs = 0; end

    % strings for mean ± std (guard when fields are missing)
    if isfield(B,'test_f1_mean') && isfield(B,'test_f1_std')
        f1_str = sprintf('$%.3f pm %.3f$', B.test_f1_mean, B.test_f1_std);
    else
        f1_str = '--';
    end

    if isfield(B,'test_accuracy_mean') && isfield(B,'test_accuracy_std')
        acc_str = sprintf('$%.3f pm %.3f$', B.test_accuracy_mean, B.test_accuracy_std);
    else
        acc_str = '--';
    end

    if isfield(B,'train_f1_mean') && isfield(B,'train_f1_std')
        trn_str = sprintf('$%.3f pm %.3f$', B.train_f1_mean, B.train_f1_std);
    else
        trn_str = '--';
    end

    % NEW polygon area
    if isfield(B,'test_polygon_area_mean') && isfield(B,'test_polygon_area_std') ...
            && ~isempty(B.test_polygon_area_mean) && ~isempty(B.test_polygon_area_std)
        poly_str = sprintf('$%.3f pm %.3f$', B.test_polygon_area_mean, B.test_polygon_area_std);
    else
        poly_str = '--';
    end

    fprintf(fid_tex, '%s & %s & %s & %s & %s & %d & %d n', ...
        set_name_clean, f1_str, acc_str, trn_str, poly_str, ...
        numel(all_results{i}.configs), n_runs);
end

fprintf(fid_tex, 'hlinen');
fprintf(fid_tex, 'end{tabular}n');
fprintf(fid_tex, 'end{table}n');

fclose(fid_tex);


%% ---------------------------------------------------------------
% 9) Save workspace
% ---------------------------------------------------------------
save(sprintf('FINAL_ALL_RESULTS_%s.mat', timestamp_str), ...
    'all_results', 'overall_best_config', 'overall_best_set');
fprintf('n  ✓ Final workspace savedn');

%% ---------------------------------------------------------------
% 10) Console wrap-up
% ---------------------------------------------------------------
fprintf('n');
fprintf('========================================n');
fprintf('    ANALYSIS COMPLETEn');
fprintf('========================================n');
fprintf('Generated filesn');
fprintf('  • publication figures (PNG + FIG)n');
fprintf('  • comprehensive text summaryn');
fprintf('  • LaTeX table filen');
fprintf('  • CSV export for all configurationsn');
fprintf('  • merged MATLAB workspacen');
fprintf('nKey Findingsn');
fprintf('  Best Parameter Set %sn', overall_best_set);
fprintf('  Best Polygon Area %.4f ± %.4fn', ...
    overall_best_config.test_polygon_area_mean, overall_best_config.test_polygon_area_std);
fprintf('  Best Test F1 %.4f ± %.4fn', overall_best_config.test_f1_mean, overall_best_config.test_f1_std);
fprintf('  Best Test Accuracy %.4f ± %.4fn', overall_best_config.test_accuracy_mean, overall_best_config.test_accuracy_std);
fprintf('nTotal Experimental Workn');
fprintf('  Configurations %dn', total_configs);
fprintf('  Total Runs %dn', total_runs_all);
fprintf('========================================n');



%% ---------------------------------------------------------------
% 12) Create & save publication tables (CSV + Excel + LaTeX)
% ---------------------------------------------------------------
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
poly_m     = zeros(nSets,1);
poly_s     = zeros(nSets,1);
cv_pct     = zeros(nSets,1);
nConfigs   = zeros(nSets,1);
nRuns      = zeros(nSets,1);

T_best = nan(nSets,1); E_best = nan(nSets,1); S_best = nan(nSets,1);
MU_best = nan(nSets,1); LA_best = nan(nSets,1); TR_best = nan(nSets,1);

for i = 1:nSets
    S = all_results{i};
    B = S.best_config;

    names(i)     = string(strrep(S.name,'_',' '));
    testF1_m(i)  = B.test_f1_mean;   testF1_s(i) = B.test_f1_std;
    testAcc_m(i) = B.test_accuracy_mean; testAcc_s(i) = B.test_accuracy_std;
    trainF1_m(i) = B.train_f1_mean;  trainF1_s(i) = B.train_f1_std;

    [poly_m(i), poly_s(i)] = best_polygon(B);
    cv_pct(i)    = 100 * (B.test_f1_std  /max(eps, B.test_f1_mean));
    nConfigs(i)  = numel(S.configs);
    nRuns(i)     = getRunsSafe(B, S);

    if isfield(B,'n_trees'),              T_best(i)  = B.n_trees; end
    if isfield(B,'n_extracted_features'), E_best(i)  = B.n_extracted_features; end
    if isfield(B,'k_selected_features'),  S_best(i)  = B.k_selected_features; end
    if isfield(B,'mu_numbers'),           MU_best(i) = B.mu_numbers; end
    if isfield(B,'lambda_numbers'),       LA_best(i) = B.lambda_numbers; end
    if isfield(B,'test_ratio'),           TR_best(i) = B.test_ratio; end
end

% Table A Best summary (with polygon)
TblBest = table( ...
    names, ...
    testF1_m, testF1_s, ...
    testAcc_m, testAcc_s, ...
    trainF1_m, trainF1_s, ...
    poly_m, poly_s, ...
    cv_pct, nConfigs, nRuns, ...
    'VariableNames', [ParameterSet,TestF1_Mean,TestF1_Std, ...
                      TestAcc_Mean,TestAcc_Std, ...
                      TrainF1_Mean,TrainF1_Std, ...
                      PolyArea_Mean,PolyArea_Std, ...
                      CV_percent,Configs,Runs]);

% Table B Rank by Test F1
[~, ord] = sort(testF1_m, 'descend');
TblRank = table( (1:nSets).', names(ord).', testF1_m(ord).', cv_pct(ord).', ...
    'VariableNames', {'Rank','ParameterSet','TestF1_Mean','CV_percent'});
% Table C Best hyperparameters
TblHyper = table( ...
    names, T_best, E_best, S_best, MU_best, LA_best, TR_best, ...
    'VariableNames', [ParameterSet,Trees,ExtractedFeat,SelectedFeat,Mu,Lambda,TestRatio]);

% Table D Experiment scale
TblScale = table(sum(nConfigs), sum(nRuns(~isnan(nRuns))), ...
    'VariableNames', [TotalConfigurations,TotalRuns]);

% Save CSV + Excel
writetable(TblBest,  fullfile(outdir, sprintf('BestConfigSummary_%s.csv', timestamp_str)));
writetable(TblRank,  fullfile(outdir, sprintf('Ranking_%s.csv',           timestamp_str)));
writetable(TblHyper, fullfile(outdir, sprintf('BestHyperparams_%s.csv',   timestamp_str)));
writetable(TblScale, fullfile(outdir, sprintf('ExperimentScale_%s.csv',   timestamp_str)));

xlsx = fullfile(outdir, sprintf('ResultsTables_%s.xlsx', timestamp_str));
writetable(TblBest,  xlsx, 'Sheet','BestSummary');
writetable(TblRank,  xlsx, 'Sheet','Ranking','WriteMode','overwrite');
writetable(TblHyper, xlsx, 'Sheet','BestHyperparams','WriteMode','overwrite');
writetable(TblScale, xlsx, 'Sheet','ExperimentScale','WriteMode','overwrite');

fprintf('  ✓ Tables saved to %s (CSV + Excel)n', outdir);

% LaTeX table writers
texA = fullfile(outdir, sprintf('BestConfigSummary_%s.tex', timestamp_str));
texB = fullfile(outdir, sprintf('Ranking_%s.tex',           timestamp_str));
texC = fullfile(outdir, sprintf('BestHyperparams_%s.tex',   timestamp_str));

latex_write_best(texA, TblBest, ...
  'Parameter Optimization Results — Best Configuration per Set', 'tabbest_summary');
latex_write_rank(texB, TblRank, ...
  'Ranking by Test F1 (best configuration per set)', 'tabranking');
latex_write_hyper(texC, TblHyper, ...
  'Winning hyperparameters per parameter set', 'tabbest_hyper');

fprintf('  ✓ LaTeX tables written to %sn', outdir);

%% ===================== LOCAL HELPERS =====================
function n = getRunsSafe(cfg, S)
% Robustly determine number of runs for a best_config.
    if isstruct(cfg) && isfield(cfg,'total_runs_completed') && ~isempty(cfg.total_runs_completed)
        n = double(cfg.total_runs_completed); return;
    end
    if isfield(cfg,'all_runs')
        if isfield(cfg.all_runs,'test_f1') && ~isempty(cfg.all_runs.test_f1)
            n = numel(cfg.all_runs.test_f1); return;
        elseif isfield(cfg.all_runs,'test_accuracy') && ~isempty(cfg.all_runs.test_accuracy)
            n = numel(cfg.all_runs.test_accuracy); return;
        end
    end
    if nargin==2 && isfield(S,'configs') && ~isempty(S.configs) ...
            && isfield(S.configs(1),'all_runs') && isfield(S.configs(1).all_runs,'test_f1')
        n = numel(S.configs(1).all_runs.test_f1); return;
    end
    n = NaN;
end

function total = count_total_runs(all_results)
% Robustly sum runs over every configuration in every set.
    total = 0;
    for i = 1numel(all_results)
        S = all_results{i};
        for c = 1numel(S.configs)
            cfg = S.configs(c);
            if isfield(cfg,'total_runs_completed') && ~isempty(cfg.total_runs_completed)
                total = total + double(cfg.total_runs_completed);
            elseif isfield(cfg,'all_runs') && isfield(cfg.all_runs,'test_f1') && ~isempty(cfg.all_runs.test_f1)
                total = total + numel(cfg.all_runs.test_f1);
            elseif isfield(cfg,'all_runs') && isfield(cfg.all_runs,'test_accuracy') && ~isempty(cfg.all_runs.test_accuracy)
                total = total + numel(cfg.all_runs.test_accuracy);
            end
        end
    end
%     % fallback if still 0 infer per-set
%     if ~isfinite(total) && total==0
%         for i = 1numel(all_results)
%             S = all_results{i};
%             if isfield(S,'configs') && ~isempty(S.configs)
%                 runs_per = NaN;
%                 cfg1 = S.configs(1);
%                 if isfield(cfg1,'total_runs_completed') && ~isempty(cfg1.total_runs_completed)
%                     runs_per = double(cfg1.total_runs_completed);
%                 elseif isfield(cfg1,'all_runs') && isfield(cfg1.all_runs,'test_f1') && ~isempty(cfg1.all_runs.test_f1)
%                     runs_per = numel(cfg1.all_runs.test_f1);
%                 end
%                 if ~isnan(runs_per), total = total + numel(S.configs)runs_per; end
%             end
%         end
%     end
end

function m = safe_mean(all_runs, field)
    if isfield(all_runs, field) && ~isempty(all_runs.(field))
        v = all_runs.(field); m = mean(v);
    else
        m = NaN;
    end
end

function s = safe_std(all_runs, field)
    if isfield(all_runs, field) && ~isempty(all_runs.(field))
        v = all_runs.(field); s = std(v);
    else
        s = NaN;
    end
end

function [m,s] = best_polygon(B)
% Return polygon area meanstd for a best_config (robust to older files)
    if isfield(B,'test_polygon_area_mean') && isfield(B,'test_polygon_area_std') ...
            && ~isempty(B.test_polygon_area_mean)
        m = B.test_polygon_area_mean; s = B.test_polygon_area_std; return;
    end
    if isfield(B,'all_runs') && isfield(B.all_runs,'test_polygon_area') ...
            && ~isempty(B.all_runs.test_polygon_area)
        m = mean(B.all_runs.test_polygon_area); s = std(B.all_runs.test_polygon_area); return;
    end
    m = NaN; s = NaN;
end

function latex_write_best(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires usepackage{booktabs}usepackage{siunitx}n');
fprintf(fid, 'begin{table}[ht]centeringn');
fprintf(fid, 'caption{%s}label{%s}n', caption, label);
fprintf(fid, ['begin{tabular}{l' ...
              ' S[table-format=1.3(3)]' ... % Test F1
              ' S[table-format=1.3(3)]' ... % Test Acc
              ' S[table-format=1.3(3)]' ... % Train F1
              ' S[table-format=1.3(3)]' ... % Polygon Area
              ' S[table-format=2.2]'   ...  % CV%
              ' S[table-format=2.0]'   ...  % Configs
              ' S[table-format=4.0]}n']); % Runs
fprintf(fid, 'toprulen');
fprintf(fid, ['textbf{Parameter Set} & {textbf{Test F1}} & {textbf{Test Acc}} & ' ...
              '{textbf{Train F1}} & {textbf{Polygon Area}} & {textbf{CV%%}} & {textbf{Configs}} & {textbf{Runs}} n']);
fprintf(fid, 'midrulen');
for i = 1height(T)
    nm = T.ParameterSet(i);
    f1 = sprintf('%.3f pm %.3f', T.TestF1_Mean(i),  T.TestF1_Std(i));
    ac = sprintf('%.3f pm %.3f', T.TestAcc_Mean(i), T.TestAcc_Std(i));
    tr = sprintf('%.3f pm %.3f', T.TrainF1_Mean(i), T.TrainF1_Std(i));
    pa = sprintf('%.3f pm %.3f', T.PolyArea_Mean(i), T.PolyArea_Std(i));
    fprintf(fid, '%s & %s & %s & %s & %s & %.2f & %d & %d n', ...
        nm, f1, ac, tr, pa, T.CV_percent(i), T.Configs(i), T.Runs(i));
end
fprintf(fid, 'bottomrulenend{tabular}nend{table}n');
fclose(fid);
end

function latex_write_rank(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires usepackage{booktabs}usepackage{siunitx}n');
fprintf(fid, 'begin{table}[ht]centeringn');
fprintf(fid, 'caption{%s}label{%s}n', caption, label);
fprintf(fid, 'begin{tabular}{c l S[table-format=1.3] S[table-format=2.2]}n');
fprintf(fid, 'toprulen');
fprintf(fid, 'textbf{Rank} & textbf{Parameter Set} & {textbf{Test F1}} & {textbf{CV%%}} n');
fprintf(fid, 'midrulen');
for i = 1height(T)
   fprintf(fid, '%d & %s & %.3f & %.2f n', T.Rank(i), T.ParameterSet(i), T.TestF1_Mean(i), T.CV_percent(i));
end
fprintf(fid, 'bottomrulenend{tabular}nend{table}n');
fclose(fid);
end

function latex_write_hyper(fname, T, caption, label)
fid = fopen(fname,'w');
fprintf(fid, '%% Requires usepackage{booktabs}usepackage{siunitx}n');
fprintf(fid, 'begin{table}[ht]centeringn');
fprintf(fid, 'caption{%s}label{%s}n', caption, label);
fprintf(fid, 'begin{tabular}{l S[table-format=3.0] S[table-format=2.0] S[table-format=2.0] S[table-format=3.0] S[table-format=3.0] S[table-format=1.2]}n');
fprintf(fid, 'toprulen');
fprintf(fid, ['textbf{Parameter Set} & textbf{Trees} & textbf{Extr.} & textbf{Sel.} & ' ...
              'boldmath{$mu$} & boldmath{$lambda$} & textbf{TestRatio} n']);
fprintf(fid, 'midrulen');
for i = 1height(T)
    fprintf(fid, '%s & %d & %d & %d & %d & %d & %.2f n', ...
        T.ParameterSet(i), T.Trees(i), T.ExtractedFeat(i), T.SelectedFeat(i), ...
        T.Mu(i), T.Lambda(i), T.TestRatio(i));
end
fprintf(fid, 'bottomrulenend{tabular}nend{table}n');
fclose(fid);
end
