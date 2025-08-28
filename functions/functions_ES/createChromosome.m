function chromosome = createChromosome(n_extracted_features, feature_pool_size, n_fuzzy_terms)
    
    % Creates a chromosome for GA-optimized fuzzy feature extraction
    % it makes a collection of parameters that defines how to transform original EEG features into new
    
    % Linear
    linear_weights = randn(n_extracted_features, feature_pool_size) * 0.1;
    
    % Nonlinear
    % Exponential: tanh(a*x + b)
    nonlinear_exp = randn(n_extracted_features, 2) * 0.5;
    
    % Sinusoidal: sin(freq*x + phase)
    nonlinear_sin = [rand(n_extracted_features,1)*2*pi, rand(n_extracted_features,1)*2*pi - pi];
    
    % Logarithmic: sign(x) * log(|c*x| + 1) ; c > 0
    nonlinear_log = rand(n_extracted_features,1)*2 + 0.1;
    
    % Power: sign(x) * |x|^p ; p ∈ [0.5, 2]
    nonlinear_pow = rand(n_extracted_features,1)*1.5 + 0.5;
    
    % Combination weights (will be normalized using softmax)
    raw_comb_weights = randn(n_extracted_features, 5);
    
    % normalizaion
    comb_weights = zeros(size(raw_comb_weights));
    for i = 1:n_extracted_features
        w = raw_comb_weights(i,:) - max(raw_comb_weights(i,:)); % Prevent overflow
        exp_w = exp(w);
        comb_weights(i,:) = exp_w / sum(exp_w);
    end
    
    fuzzy_params = zeros(n_extracted_features, n_fuzzy_terms, 3);
    for f = 1:n_extracted_features

        % after features are z-score normalized, range is [-3, 3]
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
    
    chromosome = struct();
    chromosome.linear_weights = linear_weights;
    chromosome.nonlinear_exp = nonlinear_exp;
    chromosome.nonlinear_sin = nonlinear_sin;
    chromosome.nonlinear_log = nonlinear_log;
    chromosome.nonlinear_pow = nonlinear_pow;
    chromosome.comb_weights = comb_weights;
    chromosome.fuzzy_params = fuzzy_params;
end