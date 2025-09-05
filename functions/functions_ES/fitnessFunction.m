function fitness_values = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms, train_idx)
    
    if nargin < 6 || isempty(train_idx)
        train_idx = 1:size(feature_matrix, 1);
    end
    
    % training data for fitness evaluation
    feature_matrix_train = feature_matrix(train_idx, :);
    labels_train = labels(train_idx);
    
    n_train_samples = length(train_idx);
    k_folds = 5; % Within training data
    
    unique_classes = unique(labels_train);
    
    class_0_idx = find(labels_train == 0);
    class_1_idx = find(labels_train == 1);
    
    class_0_idx = class_0_idx(randperm(length(class_0_idx)));
    class_1_idx = class_1_idx(randperm(length(class_1_idx)));
    
    n_class_0 = length(class_0_idx);
    n_class_1 = length(class_1_idx);
    
    fold_size_0 = floor(n_class_0 / k_folds);
    fold_size_1 = floor(n_class_1 / k_folds);
    
    cv_indices = cell(k_folds, 2);
    
    for fold = 1:k_folds
        if fold < k_folds
            val_0_start = (fold-1) * fold_size_0 + 1;
            val_0_end = fold * fold_size_0;
            val_1_start = (fold-1) * fold_size_1 + 1;
            val_1_end = fold * fold_size_1;
        else
            val_0_start = (fold-1) * fold_size_0 + 1;
            val_0_end = n_class_0;
            val_1_start = (fold-1) * fold_size_1 + 1;
            val_1_end = n_class_1;
        end
        
        val_0_idx = class_0_idx(val_0_start:val_0_end);
        val_1_idx = class_1_idx(val_1_start:val_1_end);
        val_fold_idx = [val_0_idx; val_1_idx];
        
        train_fold_idx = setdiff(1:n_train_samples, val_fold_idx);
        
        cv_indices{fold, 1} = train_fold_idx;
        cv_indices{fold, 2} = val_fold_idx;
    end
    
    fitness_values = zeros(length(population), 1);
    
    % Parallel
    parfor i = 1:length(population)
        try
            % Apply chromosome to training data only
            fuzzy_features = applyChromosome(population{i}, feature_matrix_train, n_extracted_features, n_fuzzy_terms);
            
            if any(isnan(fuzzy_features(:))) || any(isinf(fuzzy_features(:)))
                fitness_values(i) = 0;
                continue;
            end
            
            valid_features = var(fuzzy_features) > 1e-8;
            if sum(valid_features) < 2
                fitness_values(i) = 0;
                continue;
            end
            fuzzy_features = fuzzy_features(:, valid_features);
            
            cv_scores = zeros(k_folds, 1);
            
            for fold = 1:k_folds
                train_fold_idx = cv_indices{fold, 1};
                val_fold_idx = cv_indices{fold, 2};
                
                X_fold_train = fuzzy_features(train_fold_idx, :);
                y_fold_train = labels_train(train_fold_idx);
                X_fold_val = fuzzy_features(val_fold_idx, :);
                y_fold_val = labels_train(val_fold_idx);
                
                % ensure both classes are present in training fold
                if length(unique(y_fold_train)) < 2
                    cv_scores(fold) = 0;
                    continue;
                end
                
                % Standardize
                mu_fold = mean(X_fold_train);
                sigma_fold = std(X_fold_train);
                sigma_fold(sigma_fold < 1e-8) = 1;
                
                X_fold_train_std = (X_fold_train - mu_fold) ./ sigma_fold;
                X_fold_val_std = (X_fold_val - mu_fold) ./ sigma_fold;
                
                try
                    model = fitclinear(X_fold_train_std, y_fold_train, ...
                        'Learner', 'svm', ...
                        'Regularization', 'ridge', ...
                        'Lambda', 1e-3, ...
                        'Solver', 'lbfgs');
                    
                    y_pred = predict(model, X_fold_val_std);
                    
                    % predictions
                    if iscell(y_pred)
                        y_pred = cellfun(@str2double, y_pred);
                    end
                    
                    nan_mask = isnan(y_pred);
                    if any(nan_mask)
                        y_pred(nan_mask) = mode(y_fold_train);
                    end
                    
                    cv_scores(fold) = calculateF1Score(y_fold_val, y_pred);
                    
                catch
                    cv_scores(fold) = 0;
                end
            end
            
            valid_folds = sum(cv_scores > 0);
            if valid_folds >= 3 % At least 3 valid folds
                fitness_values(i) = mean(cv_scores(cv_scores > 0));
            else
                fitness_values(i) = 0;
            end
            
            % Penalty for too many features (>50)
            if size(fuzzy_features, 2) > 50
                fitness_values(i) = fitness_values(i) * 0.95;
            end
            
        catch
            fitness_values(i) = 0;
        end
    end
end