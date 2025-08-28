function runEvolutionStrategy()    function [best_chromosome, best_fitness_history] = evolutionaryStrategy_EEG(feature_matrix, labels, mi, l, T_max, selection_mode, n_extracted_features, feature_pool_size, n_fuzzy_terms)
    % Evolution Strategy for EEG GA-Fuzzy Feature Extraction
    % Inputs:
    %   feature_matrix - normalized EEG feature matrix [samples x features]
    %   labels - binary classification labels (0=closed, 1=opened)
    %   mi - population size (μ)
    %   l - offspring size (λ)
    %   T_max - maximum generations
    %   selection_mode - 'mi_lambda' or 'mi_plus_lambda'
    %   n_extracted_features - number of features to extract
    %   feature_pool_size - total available features
    %   n_fuzzy_terms - number of fuzzy membership terms
    %
    % Outputs:
    %   best_chromosome - best evolved chromosome
    %   best_fitness_history - fitness evolution over generations
    
    fprintf('=== EEG Evolution Strategy Started ===\n');
    fprintf('Selection mode: %s\n', selection_mode);
    fprintf('Population size μ = %d, Offspring size λ = %d\n', mi, l);
    fprintf('Features to extract: %d from %d available\n', n_extracted_features, feature_pool_size);
    fprintf('Maximum generations: %d\n', T_max);
    fprintf('Fuzzy terms: %d\n', n_fuzzy_terms);
    
    % Initialize population
    population = initialPopulation(mi, n_extracted_features, feature_pool_size, n_fuzzy_terms);
    best_fitness_history = zeros(T_max, 1);
    
    % Evaluate initial population
    fitness_parents = fitnessFunction(population, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);
    
    prev_best_fitness = [];
    convergence_threshold = 1e-5;
    
    for generation = 1:T_max
        fprintf('\n     Generation %d/%d    \n', generation, T_max);
        
        % Track fitness statistics
        current_best_fitness = max(fitness_parents);
        current_avg_fitness = mean(fitness_parents);
        fitness_std = std(fitness_parents);
        best_fitness_history(generation) = current_best_fitness;
        
        fprintf('Best fitness: %.4f\n', current_best_fitness);
        fprintf('Average fitness: %.4f\n', current_avg_fitness);
        fprintf('Fitness std: %.4f\n', fitness_std);
        
        % Check for population convergence
        if fitness_std < 1e-6
            fprintf('\nPopulation converged after %d generations\n', generation);
            fprintf('Fitness standard deviation: %.2e\n', fitness_std);
            best_fitness_history = best_fitness_history(1:generation);
            break;
        end
        
        % Check for fitness stagnation
        if ~isempty(prev_best_fitness)
            delta = abs(prev_best_fitness - current_best_fitness);
            if delta < convergence_threshold && fitness_std < 1e-3
                fprintf('\nFitness stagnated AND population converged after %d generations\n', generation);
                fprintf('Final fitness delta: %.2e\n', delta);
                fprintf('Final fitness std: %.4f\n', fitness_std);
                best_fitness_history = best_fitness_history(1:generation);
                break;
            end
        end
        prev_best_fitness = current_best_fitness;
        
        % Generate offspring using tournament selection + mutation
        parents_for_mating = matingPool(population, fitness_parents, l);
        offspring = mutation(parents_for_mating, n_extracted_features, feature_pool_size, n_fuzzy_terms);
        
        % crossover!!!
        
        % Evaluate offspring
        fitness_offspring = fitnessFunction(offspring, feature_matrix, labels, n_extracted_features, n_fuzzy_terms);
        
        fprintf('Offspring best fitness: %.4f\n', max(fitness_offspring));
        fprintf('Offspring average fitness: %.4f\n', mean(fitness_offspring));
        
        % Selection for next generation
        if strcmp(selection_mode, 'mi_lambda')
            [population, fitness_parents] = selection_mi_lambda(offspring, fitness_offspring, mi);
        elseif strcmp(selection_mode, 'mi_plus_lambda')
            [population, fitness_parents] = selection_mi_plus_lambda(population, fitness_parents, offspring, fitness_offspring, mi);
        else
            error('selection_mode must be ''mi_lambda'' or ''mi_plus_lambda''');
        end
    end
    
    % Return best solution
    [~, best_idx] = max(fitness_parents);
    best_chromosome = population{best_idx};
    
    fprintf('\n=== Evolution Strategy Completed ===\n');
    fprintf('Best fitness achieved: %.4f\n', max(best_fitness_history));
    fprintf('Generations completed: %d\n', length(best_fitness_history));
end
end