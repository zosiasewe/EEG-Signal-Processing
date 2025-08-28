function population = initialPopulation(mi, n_extracted_features, feature_pool_size, n_fuzzy_terms)

    population = cell(mi, 1);
    for i = 1:mi
        population{i} = createChromosome(n_extracted_features, feature_pool_size, n_fuzzy_terms);
    end
    
    fprintf('Created initial population of %d chromosomes\n', mi);
end