function f1_score = calculateF1Score(y_true, y_pred)
    
    % Convert to binary (0,1)
    y_true_binary = (y_true > 0.5);
    y_pred_binary = (y_pred > 0.5);
    
    % confusion matrix components
    TP = sum(y_true_binary == 1 & y_pred_binary == 1);
    FP = sum(y_true_binary == 0 & y_pred_binary == 1);
    FN = sum(y_true_binary == 1 & y_pred_binary == 0);
    
    % precision and recall
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    % F1-score
    if (precision + recall) > 0
        f1_score = 2 * (precision * recall) / (precision + recall);
    else
        f1_score = 0;
    end
end