function fitness_values = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms)
    
    fitness_values = zeros(length(population), 1);
    
    n_samples = size(feature_matrix, 1);
    k_folds = 3;
    fold_size = floor(n_samples / k_folds);
    
    shuffle_idx = randperm(n_samples);
    
    for i = 1:length(population)
        try
            % Apply chromosome to get fuzzy features
            fuzzy_features = applyChromosome(population{i}, feature_matrix, n_extracted_features, n_fuzzy_terms);
            
            if any(isnan(fuzzy_features(:))) || any(isinf(fuzzy_features(:)))
                fitness_values(i) = 0.0;
                continue;
            end
            
            % Remove features with zero variance
            feature_var = var(fuzzy_features);
            valid_features = feature_var > 1e-8;
            if sum(valid_features) < 2
                fitness_values(i) = 0.0;
                continue;
            end
            fuzzy_features = fuzzy_features(:, valid_features);
            
            % cross-validation
            cv_scores = zeros(k_folds, 1);
            
            for fold = 1:k_folds
                % test indices for this fold
                if fold < k_folds
                    test_start = (fold-1) * fold_size + 1;
                    test_end = fold * fold_size;
                else
                    test_start = (fold-1) * fold_size + 1;
                    test_end = n_samples;
                end
                
                test_idx = shuffle_idx(test_start:test_end);
                train_idx = shuffle_idx(setdiff(1:n_samples, test_start:test_end));
                
                X_train = fuzzy_features(train_idx, :);
                y_train = labels(train_idx);
                X_test = fuzzy_features(test_idx, :);
                y_test = labels(test_idx);
                
                % Standardize features for SVM
                mu_train = mean(X_train);
                sigma_train = std(X_train);
                sigma_train(sigma_train < 1e-8) = 1; % prevent division by zero
                
                X_train_std = (X_train - mu_train) ./ sigma_train;
                X_test_std = (X_test - mu_train) ./ sigma_train;
                
                try
                    % SVM with linear kernel
                    model = fitcsvm(X_train_std, y_train, ...
                        'KernelFunction', 'linear', ...
                        'Standardize', false, ...
                        'BoxConstraint', 1.0, ...
                        'Solver', 'SMO');
                    
                    y_pred = predict(model, X_test_std);
                    cv_scores(fold) = calculateF1Score(y_test, y_pred);
                    
                catch svm_error
                    % LDA if SVM fails
                    try
                        model = fitcdiscr(X_train_std, y_train, 'DiscrimType', 'linear');
                        y_pred = predict(model, X_test_std);
                        cv_scores(fold) = calculateF1Score(y_test, y_pred);
                    catch lda_error
                        % simple logistic regression equivalent
                        try
                            model = fitglm(X_train_std, y_train, 'Distribution', 'binomial');
                            y_pred_prob = predict(model, X_test_std);
                            y_pred = y_pred_prob > 0.5;
                            cv_scores(fold) = calculateF1Score(y_test, double(y_pred));
                        catch
                            cv_scores(fold) = 0.0;
                        end
                    end
                end
            end
            
            fitness_values(i) = mean(cv_scores);
            
        catch main_error
            fitness_values(i) = 0.0;
        end
    end
    
    n_features = size(fuzzy_features, 2);
    if n_features > 50 % Adjust threshold as needed
        parsimony_penalty = 0.95; % 5% penalty
        fitness_values = fitness_values * parsimony_penalty;
    end
end