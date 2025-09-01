function mutants = mutation(population, n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % Mutation function for chromosomes
    % structure: {linear_weights, nonlinear_exp, nonlinear_sin, nonlinear_log, nonlinear_pow, comb_weights, fuzzy_params}
    
    mutants = cell(length(population), 1);
    n_params = 6; % num of parameter types
    tau_1 = 1.0 / sqrt(2 * n_params);
    tau_2 = 1.0 / sqrt(2 * sqrt(n_params));
    
    for i = 1:length(population)
        chromosome = population{i};
        mutant = cell(7, 1);
        
        % Global mutation factor
        r = exp(tau_1 * randn());
        
        % Mutate each cell
        for j = 1:7
            data = chromosome{j};
            
            if j == 1 % linear_weights
                mutation_mask = rand(size(data)) < 0.25;
                sigma = 0.03 * r * exp(tau_2 * randn());
                mutant{j} = data + sigma * randn(size(data)) .* mutation_mask;
                
            elseif j == 2 % nonlinear_exp
                mutation_mask = rand(size(data)) < 0.25;
                sigma = 0.1 * r * exp(tau_2 * randn());
                mutant{j} = data + sigma * randn(size(data)) .* mutation_mask;
                
            elseif j == 3 % nonlinear_sin
                sigma = 0.1 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                mutant_data(:, 1) = max(0.5, min(3.0, mutant_data(:, 1)));
                mutant{j} = mutant_data;
                
            elseif j == 4 % nonlinear_log
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                mutant{j} = max(0.1, min(2.0, mutant_data));
                
            elseif j == 5 % nonlinear_pow
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                mutant{j} = max(0.5, min(2.0, mutant_data));
                
            elseif j == 6 % comb_weights
                sigma = 0.2 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                % Renormalize using softmax
                for k = 1:size(mutant_data, 1)
                    w = mutant_data(k, :) - max(mutant_data(k, :));
                    exp_w = exp(w);
                    mutant_data(k, :) = exp_w / sum(exp_w);
                end
                mutant{j} = mutant_data;
                
            elseif j == 7 % fuzzy_params
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant{j} = data + sigma * randn(size(data));
            end
        end
        
        mutants{i} = mutant;
    end
end