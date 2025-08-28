function mutants = mutation(population, n_extracted_features, feature_pool_size, n_fuzzy_terms)
 
    mutants = {};
    n_params = 6; % num of parameter types in chromosome
    tau_1 = 1.0 / sqrt(2 * n_params);
    tau_2 = 1.0 / sqrt(2 * sqrt(n_params));
    
    for i = 1:length(population)
        chromosome = population{i};
        mutant = struct();
        
        % global mutation factor
        r = exp(tau_1 * randn());
        
        % mutate each field with adaptive step sizes
        fields = fieldnames(chromosome);
        
        for j = 1:length(fields)
            field = fields{j};
            data = chromosome.(field);
            
            if strcmp(field, 'linear_weights')
                % mutation for linear weights
                sigma = 0.01 * r * exp(tau_2 * randn());
                mutation_mask = rand(size(data)) < 0.1;
                mutant.(field) = data + sigma * randn(size(data)) .* mutation_mask;
                
            elseif strcmp(field, 'nonlinear_exp')
                % mutation for exponential parameters
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutation_mask = rand(size(data)) < 0.1;
                mutant.(field) = data + sigma * randn(size(data)) .* mutation_mask;
                
            elseif strcmp(field, 'nonlinear_sin')
                % mutation for sinusoidal parameters
                sigma = 0.1 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                % [0.5, 3]
                mutant_data(:, 1) = max(0.5, min(3.0, mutant_data(:, 1)));
                mutant.(field) = mutant_data;
                
            elseif strcmp(field, 'nonlinear_log')
                % mutation for logarithmic parameters
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                % [0.1, 2.0]
                mutant.(field) = max(0.1, min(2.0, mutant_data));
                
            elseif strcmp(field, 'nonlinear_pow')
                % mutation for power parameters
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                % [0.5, 2.0]
                mutant.(field) = max(0.5, min(2.0, mutant_data));
                
            elseif strcmp(field, 'comb_weights')
                % mutation for combination weights
                sigma = 0.1 * r * exp(tau_2 * randn());
                mutant_data = data + sigma * randn(size(data));
                % norm softmax
                for k = 1:size(mutant_data, 1)
                    w = mutant_data(k, :) - max(mutant_data(k, :));
                    exp_w = exp(w);
                    mutant_data(k, :) = exp_w / sum(exp_w);
                end
                mutant.(field) = mutant_data;
                
            elseif strcmp(field, 'fuzzy_params')
                % mutation for fuzzy parameters
                sigma = 0.05 * r * exp(tau_2 * randn());
                mutant.(field) = data + sigma * randn(size(data));
                
            else
                mutant.(field) = data;
            end
        end
        
        mutants{i} = mutant;
    end
end
