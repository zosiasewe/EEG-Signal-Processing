function [new_population, new_fitness] = selection_mu_lambda(offspring, fitness_offspring, mu)
    % (μ, λ) selection - select best μ from λ offspring only
    [sorted_fitness, sort_idx] = sort(fitness_offspring, 'descend');
    top_indices = sort_idx(1:mu);
    
    new_population = offspring(top_indices);
    new_fitness = sorted_fitness(1:mu);
end