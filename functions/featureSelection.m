function [selected_features, selected_indices] = featureSelection(features, labels, k)

    n_features = size(features, 2);
    scores = zeros(n_features, 1);
    
    % Pearson correlation with labels
    for i = 1:n_features
        scores(i) = abs(corr(features(:,i), labels));
    end
    
    % Sort values in descending order
    [~, sorted_idx] = sort(scores, 'descend');
    selected_indices = sorted_idx(1:k);
    selected_features = features(:, selected_indices);
    
    fprintf('Selected %d features using correlation method\n', k);
end