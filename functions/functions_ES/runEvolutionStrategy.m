function [best_chromosome, best_fitness_history] = runEvolutionStrategy(feature_matrix, labels, mu, lambda, T_max, selection_mode, n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % parallel pool if not running
    if isempty(gcp('nocreate'))
        parpool; 
    end

    population = initialPopulation(mu, n_extracted_features, feature_pool_size, n_fuzzy_terms);
    best_fitness_history = zeros(T_max,1);
    fitness_parents = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);

    prev_best_fitness = [];
    patience = 15;
    tol = 1e-5;
    no_improve_count = 0;

    for generation = 1:T_max
        tic;
        current_best_fitness = max(fitness_parents);
        current_avg_fitness  = mean(fitness_parents);
        fitness_std = std(fitness_parents);
        best_fitness_history(generation) = current_best_fitness;

        fprintf('Generation %d/%d: Best=%.4f, Avg=%.4f, Std=%.4f\n', generation, T_max, current_best_fitness, current_avg_fitness, fitness_std);

        % Early stopping
        if ~isempty(prev_best_fitness)
            if current_best_fitness - prev_best_fitness < tol
                no_improve_count = no_improve_count + 1;
            else
                no_improve_count = 0;
            end
        end
        prev_best_fitness = current_best_fitness;
        if no_improve_count >= patience
            fprintf('Early stopping: no improvement for %d generations.\n', patience);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end

        if fitness_std < 1e-5
            fprintf('Population converged after %d generations\n', generation);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end

        % Generate offspring
        parents_for_mating = matingPool(population, fitness_parents, lambda);
        crossover_offspring = crossover(parents_for_mating, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        offspring = mutation(crossover_offspring, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        fitness_offspring = fitnessFunction(offspring, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);

        fprintf('Offspring best: %.4f, Avg: %.4f\n', max(fitness_offspring), mean(fitness_offspring));

        % Selection
        if strcmp(selection_mode,'mu_lambda')
            [population, fitness_parents] = selection_mu_lambda(offspring, fitness_offspring, mu);
        else
            [population, fitness_parents] = selection_mu_plus_lambda(population, fitness_parents, offspring, fitness_offspring, mu);
        end
        fprintf('Generation time: %.2f seconds\n\n', toc);
    end

    % Return best chromosome
    [~, best_idx] = max(fitness_parents);
    best_chromosome = population{best_idx};
    fprintf('Evolution completed. Best fitness: %.4f, Generations: %d\n', max(best_fitness_history), length(best_fitness_history));
end
