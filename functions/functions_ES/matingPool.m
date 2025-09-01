function selected_ind = matingPool(population, fitness_values, num_to_select)
  
    if nargin < 3
        num_to_select = length(population); % Default to population size
    end
    
    tournament_size = 3;
    population_size = length(population);
    selected_ind = cell(num_to_select, 1);
    
    for i = 1:num_to_select 
        % Tournament selection
        tournament_idx = randperm(population_size, min(tournament_size, population_size));
        tournament_fitness = fitness_values(tournament_idx);
        [~, best_idx] = max(tournament_fitness);
        selected_ind{i} = population{tournament_idx(best_idx)};
    end
end