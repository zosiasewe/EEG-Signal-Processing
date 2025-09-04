function offspring = crossover(parents, n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % Cell structure: {linear_weights, nonlinear_exp, nonlinear_sin, nonlinear_log, nonlinear_pow, comb_weights, fuzzy_params}
    
    offspring = cell(length(parents), 1);
    crossover_rate = 0.6;
    
    for i = 1:length(parents)
        if rand() < crossover_rate && length(parents) > 1
            parent1 = parents{i};
            parent2 = parents{mod(i, length(parents)) + 1};
            
            % blend factor
            blend_factor = 0.4 + 0.2 * rand(); % 0.4-0.6
            
            child = cell(7, 1);
            
            child{1} = blend_factor * parent1{1} + (1-blend_factor) * parent2{1}; % linear_weights
            child{2} = blend_factor * parent1{2} + (1-blend_factor) * parent2{2}; % nonlinear_exp
            child{3} = blend_factor * parent1{3} + (1-blend_factor) * parent2{3}; % nonlinear_sin
            child{4} = blend_factor * parent1{4} + (1-blend_factor) * parent2{4}; % nonlinear_log
            child{5} = blend_factor * parent1{5} + (1-blend_factor) * parent2{5}; % nonlinear_pow
            
            temp_weights = blend_factor * parent1{6} + (1-blend_factor) * parent2{6};
            child{6} = temp_weights ./ sum(temp_weights, 2);
            
            child{7} = blend_factor * parent1{7} + (1-blend_factor) * parent2{7};
            
            for f = 1:n_extracted_features
                for t = 1:n_fuzzy_terms
                    vals = sort([child{7}(f,t,1), child{7}(f,t,2), child{7}(f,t,3)]);
                    child{7}(f,t,1) = vals(1);
                    child{7}(f,t,2) = vals(2);
                    child{7}(f,t,3) = vals(3);
                end
            end
            
            offspring{i} = child;
        else
            % no crossover - copy parent
            offspring{i} = parents{i};
        end
    end
end