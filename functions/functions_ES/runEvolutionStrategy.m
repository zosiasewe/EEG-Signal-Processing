function [best_chromosome, best_fitness_history] = runEvolutionStrategy(feature_matrix, labels, mu, lambda, T_max, selection_mode, n_extracted_features, feature_pool_size, n_fuzzy_terms, train_idx)
    
    if nargin < 10 || isempty(train_idx)
        train_idx = 1:length(labels);
        fprintf('No train_idx provided, using all data\n');
    end
    
    if isempty(gcp('nocreate'))
        parpool;
    end
    
    fprintf('ES using %d training samples out of %d total samples\n', length(train_idx), length(labels));
    
    population = initialPopulation(mu, n_extracted_features, feature_pool_size, n_fuzzy_terms);
    best_fitness_history = zeros(T_max,1);
    
    fitness_parents = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms, train_idx);
    
    prev_best_fitness = [];
    patience = 20; % patience
    tol = 5e-4; % tolerance
    no_improve_count = 0;
    
    % diversity
    fitness_history_window = [];
    
    for generation = 1:T_max
        tic;
        current_best_fitness = max(fitness_parents);
        current_avg_fitness = mean(fitness_parents);
        fitness_std = std(fitness_parents);
        best_fitness_history(generation) = current_best_fitness;
        
        fprintf('Generation %d/%d: Best=%.4f, Avg=%.4f, Std=%.4f\n', generation, T_max, current_best_fitness, current_avg_fitness, fitness_std);
    
        % Early stopping with trend analysis
        if ~isempty(prev_best_fitness)
            improvement = current_best_fitness - prev_best_fitness;
            if improvement < tol
                no_improve_count = no_improve_count + 1;
            else
                no_improve_count = 0;
            end
        end
        
        fitness_history_window = [fitness_history_window, current_best_fitness];
        if length(fitness_history_window) > 10
            fitness_history_window = fitness_history_window(2:end);
        end
        
        prev_best_fitness = current_best_fitness;
        
        if no_improve_count >= patience
            fprintf('Early stopping: no improvement for %d generations.\n', patience);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end
        
        if fitness_std < 1e-5 && generation > 10
            fprintf('Population converged after %d generations\n', generation);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end
    
        % Reduce lambda if fitness is stagnating to focus search
        adaptive_lambda = lambda;
        if length(fitness_history_window) >= 5
            recent_trend = mean(diff(fitness_history_window(end-4:end)));
            if abs(recent_trend) < 1e-4 
                adaptive_lambda = max(mu, round(lambda * 0.7));
            end
        end
    
        parents_for_mating = matingPool(population, fitness_parents, adaptive_lambda);
        crossover_offspring = crossover(parents_for_mating, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        offspring = mutation(crossover_offspring, n_extracted_features, feature_pool_size, n_fuzzy_terms);
    
        % Validate offspring before fitness evaluation
        valid_offspring = cell(length(offspring), 1);
        valid_count = 0;
        
        for i = 1:length(offspring)
            if validateChromosome(offspring{i})
                valid_count = valid_count + 1;
                valid_offspring{valid_count} = offspring{i};
            else
                % Replace invalid offspring with mutated parent
                valid_count = valid_count + 1;
                parent_idx = randi(length(population));
                valid_offspring{valid_count} = mutation({population{parent_idx}}, n_extracted_features, feature_pool_size, n_fuzzy_terms);
                valid_offspring{valid_count} = valid_offspring{valid_count}{1};
            end
        end
        
        valid_offspring = valid_offspring(1:valid_count);
    
        % Evaluate offspring fitness using only training data
        fitness_offspring = fitnessFunction(valid_offspring, feature_matrix, labels, n_extracted_features, n_fuzzy_terms, train_idx);
        
        fprintf('Offspring best: %.4f, Avg: %.4f\n', max(fitness_offspring), mean(fitness_offspring));
    
        if strcmp(selection_mode,'mu_lambda')
            [population, fitness_parents] = selection_mu_lambda(valid_offspring, fitness_offspring, mu);
        else
            [population, fitness_parents] = selection_mu_plus_lambda(population, fitness_parents, valid_offspring, fitness_offspring, mu);
        end
        
        % if population becomes too uniform
        if fitness_std < 1e-4 && generation < T_max * 0.8
            n_inject = max(1, round(mu * 0.1)); % Inject 10% new individuals
            for inj = 1:n_inject
                worst_idx = find(fitness_parents == min(fitness_parents), 1);
                population{worst_idx} = createChromosome(n_extracted_features, feature_pool_size, n_fuzzy_terms);
                fitness_parents(worst_idx) = fitnessFunction({population{worst_idx}}, feature_matrix, labels, n_extracted_features, n_fuzzy_terms, train_idx);
            end
            fprintf('Injected %d new individuals for diversity\n', n_inject);
        end
    
        fprintf('Generation time: %.2f seconds\n\n', toc);
    end
    
    % best chromosome
    [~, best_idx] = max(fitness_parents);
    best_chromosome = population{best_idx};
    
    fprintf('Evolution completed. Best fitness: %.4f, Generations: %d\n', max(best_fitness_history), length(best_fitness_history));
    fprintf('Fitness evaluated using %d training samples only\n', length(train_idx));
end