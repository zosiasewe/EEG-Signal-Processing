function fitness_values = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms)
    n_samples = size(feature_matrix,1);  % 1600 (800 closed + 800 open)
    k_folds = 5; % +/-320 samples 

    % Precompute CV splits
    shuffle_idx = randperm(n_samples);
    cv_indices = cell(k_folds,2);
    fold_size = floor(n_samples/k_folds);
    for fold = 1:k_folds
        if fold < k_folds
            test_start = (fold-1)*fold_size + 1;
            test_end   = fold*fold_size;
        else % Last fold gets remaining samples
            test_start = (fold-1)*fold_size + 1;
            test_end   = n_samples;
        end
        test_idx  = shuffle_idx(test_start:test_end);
        train_idx = shuffle_idx(setdiff(1:n_samples, test_start:test_end));
        cv_indices{fold,1} = train_idx;
        cv_indices{fold,2} = test_idx;
    end

    fitness_values = zeros(length(population),1);

    % Parallel evaluation over chromosomes
    parfor i = 1:length(population)
        try
            fuzzy_features = applyChromosome(population{i}, feature_matrix, n_extracted_features, n_fuzzy_terms);

            if any(isnan(fuzzy_features(:))) || any(isinf(fuzzy_features(:))) %NaaN
                fitness_values(i) = 0;
                continue;
            end

            valid_features = var(fuzzy_features) > 1e-8; % too few valid features
            if sum(valid_features) < 2
                fitness_values(i) = 0;
                continue;
            end
            fuzzy_features = fuzzy_features(:, valid_features);

            cv_scores = zeros(k_folds,1);

            for fold = 1:k_folds
                train_idx = cv_indices{fold,1};
                test_idx  = cv_indices{fold,2};

                X_train = fuzzy_features(train_idx,:);
                y_train = labels(train_idx);
                X_test  = fuzzy_features(test_idx,:);
                y_test  = labels(test_idx);

                mu_train = mean(X_train);
                sigma_train = std(X_train);
                sigma_train(sigma_train<1e-8) = 1;

                X_train_std = (X_train - mu_train)./sigma_train;
                X_test_std  = (X_test - mu_train)./sigma_train;

                try 
                    %SVM
                    model = fitclinear(X_train_std, y_train, ...
                        'Learner','svm', ...
                        'Regularization','ridge', ...
                        'Lambda',1e-3, ...
                        'Solver','lbfgs');
                    y_pred = predict(model, X_test_std);
                    cv_scores(fold) = calculateF1Score(y_test, y_pred);
                catch
                    cv_scores(fold) = 0;
                end
            end

            fitness_values(i) = mean(cv_scores);

            %  if chromosome produces too many features (>50) we reduce its fitness (here 5%)
            if size(fuzzy_features,2) > 50
                fitness_values(i) = fitness_values(i)*0.95;
            end

        catch
            fitness_values(i) = 0;
        end
    end
end
