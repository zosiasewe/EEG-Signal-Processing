function chromosome = createChromosome(n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % structure: {linear_weights, nonlinear_exp, nonlinear_sin, nonlinear_log, nonlinear_pow, comb_weights, fuzzy_params}
    
    chromosome = cell(7, 1);
    
    % Linear weights
    chromosome{1} = randn(n_extracted_features, feature_pool_size) * 0.1;
    
    % Nonlinear parameters
    chromosome{2} = randn(n_extracted_features, 2) * 0.5; % nonlinear_exp
    chromosome{3} = [rand(n_extracted_features,1)*2*pi, rand(n_extracted_features,1)*2*pi - pi]; % nonlinear_sin
    chromosome{4} = rand(n_extracted_features,1)*2 + 0.1; % nonlinear_log
    chromosome{5} = rand(n_extracted_features,1)*1.5 + 0.5; % nonlinear_pow
    
    % Combination weights
    raw_comb_weights = randn(n_extracted_features, 5);
    comb_weights = zeros(size(raw_comb_weights));
    for i = 1:n_extracted_features
        w = raw_comb_weights(i,:) - max(raw_comb_weights(i,:));
        exp_w = exp(w);
        comb_weights(i,:) = exp_w / sum(exp_w);
    end
    chromosome{6} = comb_weights;
    
    % Fuzzy parameters
    fuzzy_params = zeros(n_extracted_features, n_fuzzy_terms, 3);
    for f = 1:n_extracted_features
        range_start = -3;
        range_end = 3;
        total_range = range_end - range_start;
        overlap = 0.3;
        step_size = total_range / (n_fuzzy_terms - 1 + 2*overlap);
        
        for t = 1:n_fuzzy_terms
            center = range_start + (t-1) * step_size + overlap * step_size;
            width = step_size * (1 + overlap);
            
            fuzzy_params(f, t, 1) = center - width/2;  % a (left point)
            fuzzy_params(f, t, 2) = center;            % b (center point)
            fuzzy_params(f, t, 3) = center + width/2;  % c (right point)
        end
    end
    chromosome{7} = fuzzy_params;
end