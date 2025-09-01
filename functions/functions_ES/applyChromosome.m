function fuzzy_features = applyChromosome(chromosome, feature_matrix_normalized, n_extracted_features, n_fuzzy_terms)
    % Apply chromosome to create features
    % structure: {linear_weights, nonlinear_exp, nonlinear_sin, nonlinear_log, nonlinear_pow, comb_weights, fuzzy_params}
    
    n_trials = size(feature_matrix_normalized, 1);
    
    % Extract chromosome components from cell array
    linear_weights = chromosome{1};
    nonlinear_exp = chromosome{2};
    nonlinear_sin = chromosome{3};
    nonlinear_log = chromosome{4};
    nonlinear_pow = chromosome{5};
    comb_weights = chromosome{6};
    fuzzy_params = chromosome{7};
    
    extracted_features = zeros(n_trials, n_extracted_features);
    
    for feat_idx = 1:n_extracted_features
        x = feature_matrix_normalized * linear_weights(feat_idx,:)';
        
        % Numerical stability
        x = max(-10, min(10, x));
        
        exp_comp = tanh(nonlinear_exp(feat_idx,1)*x + nonlinear_exp(feat_idx,2));
        sin_comp = sin(nonlinear_sin(feat_idx,1)*x + nonlinear_sin(feat_idx,2));
        log_comp = sign(x) .* log(abs(nonlinear_log(feat_idx)*x) + 1);
        pow_comp = sign(x) .* (abs(x).^nonlinear_pow(feat_idx));
        
        weights = comb_weights(feat_idx,:);
        extracted_features(:,feat_idx) = weights(1)*x + weights(2)*exp_comp + ...
                                        weights(3)*sin_comp + weights(4)*log_comp + ...
                                        weights(5)*pow_comp;
    end
    
    fuzzy_features = zeros(n_trials, n_extracted_features * n_fuzzy_terms);
    
    for feat_idx = 1:n_extracted_features
        feat_vector = extracted_features(:,feat_idx);
        
        for term = 1:n_fuzzy_terms
            a = fuzzy_params(feat_idx, term, 1);  % left point
            b = fuzzy_params(feat_idx, term, 2);  % center point
            c = fuzzy_params(feat_idx, term, 3);  % right point
            
            % Triangular membership function
            mu_left = (feat_vector - a) ./ (b - a + eps);
            mu_right = (c - feat_vector) ./ (c - b + eps);
            mu = max(0, min(mu_left, mu_right));
            
            fuzzy_features(:, (feat_idx-1)*n_fuzzy_terms + term) = mu;
        end
    end
end