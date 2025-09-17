function [y_pred_test, y_pred_train, rf_model] = classifyData(X, y, train_idx, test_idx, k_folds, n_trees)

    if nargin < 5
        k_folds = 5;
    end
    if nargin < 6
        n_trees = 100;
    end
    
    % Training and testing data
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    n_train = length(train_idx);
    
    y_pred_train = zeros(n_train, 1);
    cv_scores = zeros(k_folds, 1);
    
    % CV that ensures both classes in each fold
    fprintf('Performing %d-fold cross-validation on training data:\n', k_folds);
    fprintf('Training data class distribution: Class 0: %d, Class 1: %d\n', sum(y_train==0), sum(y_train==1));
    
    class_0_idx = find(y_train == 0);
    class_1_idx = find(y_train == 1);
    
    class_0_idx = class_0_idx(randperm(length(class_0_idx)));
    class_1_idx = class_1_idx(randperm(length(class_1_idx)));
    
    % samples per fold for each class
    n_class_0 = length(class_0_idx);
    n_class_1 = length(class_1_idx);
    
    fold_size_0 = floor(n_class_0 / k_folds);
    fold_size_1 = floor(n_class_1 / k_folds);
    
    fprintf('Per fold: ~%d class 0, ~%d class 1 samples\n', fold_size_0, fold_size_1);
    
    for fold = 1:k_folds
        fprintf('  Fold %d/%d: ', fold, k_folds);
        
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
        
        train_fold_idx = setdiff(1:n_train, val_fold_idx);
        
        y_fold_train = y_train(train_fold_idx);
        y_fold_val = y_train(val_fold_idx);
        
        fprintf('Train[0:%d, 1:%d] Val[0:%d, 1:%d] - ', sum(y_fold_train==0), sum(y_fold_train==1), sum(y_fold_val==0), sum(y_fold_val==1));
        
        if length(unique(y_fold_train)) < 2
            warning('Fold %d training set has only one class! Using majority prediction.', fold);
            majority_class = mode(y_train);
            y_pred_train(val_fold_idx) = majority_class;
            cv_scores(fold) = 0;
            fprintf('F1=0.0000 (single class), Acc=%.4f\n', sum(y_fold_val == majority_class) / length(y_fold_val));
            continue;
        end
        
        if length(unique(y_fold_val)) < 2
            warning('Fold %d validation set has only one class!', fold);
        end
        
        % Split training data into train/validation for this fold
        X_fold_train = X_train(train_fold_idx, :);
        X_fold_val = X_train(val_fold_idx, :);
        
        try
            mdl = fitcensemble(X_fold_train, y_fold_train, ...
                'Method', 'Bag', ...
                'NumLearningCycles', min(50, n_trees), ...
                'Learners', 'tree');
            
            % Predict on validation fold
            y_fold_pred = predict(mdl, X_fold_val);
            
            if iscell(y_fold_pred) %numeric
                y_fold_pred = cellfun(@str2double, y_fold_pred);
            end
            
            nan_mask = isnan(y_fold_pred); %nan
            if any(nan_mask)
                y_fold_pred(nan_mask) = mode(y_fold_train);
            end
            
            y_pred_train(val_fold_idx) = y_fold_pred;
            
            % F1 score
            cv_scores(fold) = calculateF1Score(y_fold_val, y_fold_pred);
            fold_accuracy = sum(y_fold_val == y_fold_pred) / length(y_fold_val);
            
            fprintf('F1=%.4f, Acc=%.4f\n', cv_scores(fold), fold_accuracy);
            
        catch ME
            warning('Model training failed on fold %d: %s', fold, ME.message);
            y_pred_train(val_fold_idx) = mode(y_train);
            cv_scores(fold) = 0;
            fprintf('F1=0.0000 (failed), Acc=%.4f\n', sum(y_fold_val == mode(y_train)) / length(y_fold_val));
        end
    end
    
    mean_cv_f1 = mean(cv_scores);
    std_cv_f1 = std(cv_scores);
    cv_accuracy = sum(y_train == y_pred_train) / length(y_train);
    
    % Train final model on entire training set using TreeBagger
    try
        rf_model = TreeBagger(n_trees, X_train, y_train, ...
            'Method', 'classification', ...
            'MinLeafSize', max(1, floor(length(y_train)/50)), ...
            'NumPredictorsToSample', max(1, floor(sqrt(size(X_train, 2)))), ...
            'Options', statset('UseParallel', true));
        
        % Predict on test set (unseen data)
        [y_pred_test_cell, ~] = predict(rf_model, X_test);
        
        if iscell(y_pred_test_cell)
            y_pred_test = cellfun(@str2double, y_pred_test_cell);
        else
            y_pred_test = y_pred_test_cell;
        end
        
        nan_mask = isnan(y_pred_test);
        if any(nan_mask)
            y_pred_test(nan_mask) = mode(y_train);
        end
        
        test_f1 = calculateF1Score(y_test, y_pred_test);
        test_accuracy = sum(y_test == y_pred_test) / length(y_test);
        
    catch ME
        error('Final model training failed: %s', ME.message);
    end
    
    fprintf('\n\n');
    fprintf('Training Set Performance (CV):\n');
    fprintf('  F1 Score: %.4f ± %.4f\n', mean_cv_f1, std_cv_f1);
    fprintf('  Accuracy: %.4f\n', cv_accuracy);
    fprintf('\nTest Set Performance (Final):\n');
    fprintf('  F1 Score: %.4f\n', test_f1);
    fprintf('  Accuracy: %.4f\n', test_accuracy);
end