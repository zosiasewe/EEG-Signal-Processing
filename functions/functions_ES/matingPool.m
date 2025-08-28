function selected_ind = matingPool(population, fitness_values, tournament_size)
    if nargin < 3
        tournament_size = 3;
    end

    population_size = numel(population); % num of individuals inside the cell array
    selected_ind = cell(1, population_size); % initialized cell array for the chosen ones

    for i = 1:population_size
        % idx for tournament
        tournament_idx = randperm(population_size, min(tournament_size, population_size));
        
        % fitness individuals in tournament
        tournament_fitness = fitness_values(tournament_idx);
        
        % the best one
        [~, best_idx] = max(tournament_fitness);
        selected_ind{i} = population{tournament_idx(best_idx)};
    end
end
