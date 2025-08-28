function fitness_values = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms)
    
    fitness_values = zeros(length(population), 1);
    
    for i = 1:length(population)
        try

            fuzzy_features = applyChromosome(population{i}, feature_matrix, n_extracted_features, n_fuzzy_terms);
            
            % invalid features
            if any(isnan(fuzzy_features(:))) || any(isinf(fuzzy_features(:)))
                fitness_values(i) = 0.0;
                continue;
            end
            
            % 5-fold cross-validation
            n_samples = size(fuzzy_features, 1);
            k_folds = 5;
            fold_size = floor(n_samples / k_folds);
            cv_scores = zeros(k_folds, 1);
            
            % random permutation for folds
            rng(42);
            shuffle_idx = randperm(n_samples);
            
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
                
                try
                    % Random Forest for classification
                    nTrees = 30;
                    model = TreeBagger(nTrees, X_train, y_train, ...
                        'Method', 'classification', ...
                        'OOBPrediction', 'off');
                    
                    [predictions, ~] = predict(model, X_test);
                    y_pred = str2double(predictions);
                    
                    % F1-score
                    f1_score = calculateF1Score(y_test, y_pred);
                    cv_scores(fold) = f1_score;
                    
                catch
                    % LDA if Random Forest fails
                    try
                        model = fitcdiscr(X_train, y_train);
                        y_pred = predict(model, X_test);
                        f1_score = calculateF1Score(y_test, y_pred);
                        cv_scores(fold) = f1_score;
                    catch
                        cv_scores(fold) = 0.0;
                    end
                end
            end
            
            fitness_values(i) = median(cv_scores);
            
        catch
            fitness_values(i) = 0.0;
        end
    end
end