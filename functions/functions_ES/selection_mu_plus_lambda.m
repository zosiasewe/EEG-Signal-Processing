function [new_population, new_fitness] = selection_mu_plus_lambda(parents, fitness_parents, offspring, fitness_offspring, mi)
    % (μ + λ) selection - select best μ from μ parents + λ offspring
    
    % Properly combine cell arrays
    combined_population = cell(length(parents) + length(offspring), 1);
    combined_population(1:length(parents)) = parents;
    combined_population(length(parents)+1:end) = offspring;
    
    % Combine fitness arrays
    combined_fitness = [fitness_parents; fitness_offspring];
    
    [sorted_fitness, sort_idx] = sort(combined_fitness, 'descend');
    top_indices = sort_idx(1:mi);
    
    new_population = combined_population(top_indices);
    new_fitness = sorted_fitness(1:mi);
end