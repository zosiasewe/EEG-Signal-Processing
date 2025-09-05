function isValid = validateChromosome(chromosome)
    % Basic validation of chromosome structure and values
    isValid = true;
    
    try
        if length(chromosome) ~= 7
            isValid = false;
            return;
        end
        
        for i = 1:7
            if any(~isfinite(chromosome{i}(:)))
                isValid = false;
                return;
            end
        end
        
        comb_weights = chromosome{6};
        row_sums = sum(comb_weights, 2);
        if any(abs(row_sums - 1) > 1e-6)
            isValid = false;
            return;
        end
        
        fuzzy_params = chromosome{7};
        for f = 1:size(fuzzy_params, 1)
            for t = 1:size(fuzzy_params, 2)
                a = fuzzy_params(f, t, 1);
                b = fuzzy_params(f, t, 2);
                c = fuzzy_params(f, t, 3);
                if a > b || b > c
                    isValid = false;
                    return;
                end
            end
        end
        
    catch
        isValid = false;
    end
end