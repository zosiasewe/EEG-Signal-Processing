function [best_chromosome, best_fitness_history] = runEvolutionStrategy(feature_matrix, labels, mu, lambda, T_max, selection_mode, n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % Evolution Strategy with early stopping and convergence checks
    
    fprintf('Selection mode: %s\n', selection_mode);
    fprintf('Population size μ = %d, Offspring size λ = %d\n', mu, lambda);
    fprintf('Features to extract: %d from %d available\n', n_extracted_features, feature_pool_size);
    fprintf('Maximum generations: %d\n', T_max);
    fprintf('Fuzzy terms: %d\n', n_fuzzy_terms);
    
    % Initialize population
    population = initialPopulation(mu, n_extracted_features, feature_pool_size, n_fuzzy_terms);
    best_fitness_history = zeros(T_max, 1);
    fitness_parents = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);
    
    % Early stopping
    prev_best_fitness = [];
    patience = 15;
    tol = 1e-5;
    no_improvement_count = 0;
    
    for generation = 1:T_max
        current_best_fitness = max(fitness_parents);
        current_avg_fitness = mean(fitness_parents);
        fitness_std = std(fitness_parents);
        best_fitness_history(generation) = current_best_fitness;
        
        fprintf('\n     Generation %d/%d    \n', generation, T_max);
        fprintf('Best fitness: %.4f\n', current_best_fitness);
        fprintf('Average fitness: %.4f\n', current_avg_fitness);
        fprintf('Fitness std: %.4f\n', fitness_std);
    
        % Early stopping
        if ~isempty(prev_best_fitness)
            delta = current_best_fitness - prev_best_fitness;
            if delta < tol
                no_improvement_count = no_improvement_count + 1;
            else
                no_improvement_count = 0;
            end
        end
        prev_best_fitness = current_best_fitness;
    
        if no_improvement_count >= patience
            fprintf('\nEarly stopping: no improvement for %d generations.\n', patience);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end
    
        if fitness_std < 1e-5
            fprintf('\nPopulation converged after %d generations\n', generation);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end
    
        % Offspring generation
        parents_for_mating = matingPool(population, fitness_parents, lambda);
        crossover_offspring = crossover(parents_for_mating, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        offspring = mutation(crossover_offspring, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        fitness_offspring = fitnessFunction(offspring, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);
       
        for i = 1:7
            fprintf('[%s] ', mat2str(size(offspring{1}{i})));
        end
        fprintf('\n');
    
        fprintf('Offspring best fitness: %.4f\n', max(fitness_offspring));
        fprintf('Offspring average fitness: %.4f\n', mean(fitness_offspring));
    
        % Selection
        if strcmp(selection_mode, 'mu_lambda')
            [population, fitness_parents] = selection_mu_lambda(offspring, fitness_offspring, mu);
        elseif strcmp(selection_mode, 'mu_plus_lambda')
            [population, fitness_parents] = selection_mu_plus_lambda(population, fitness_parents, offspring, fitness_offspring, mu);
        else
            error('selection_mode must be ''mu_lambda'' or ''mu_plus_lambda''');
        end
    end
    
    % Return best solution
    [~, best_idx] = max(fitness_parents);
    best_chromosome = population{best_idx};
    
    fprintf('\n    Evolution Strategy Completed     \n');
    fprintf('Best fitness achieved: %.4f\n', max(best_fitness_history));
    fprintf('Generations completed: %d\n', length(best_fitness_history));

end
