function [ica_result, W, A] = ICA(signal, n_components, iterations, epsilon, alpha)
    [m, n] = size(signal);

    % Centering
    mean_signal = mean(signal, 2);
    X_centered = signal - mean_signal;
    
    C = cov(X_centered');
    [V, D] = eig(C);
    
    [D_sorted, idx] = sort(diag(D), 'descend');
    V_sorted = V(:, idx);
    
    
    V_reduced = V_sorted(:, 1:n_components);
    D_reduced = diag(D_sorted(1:n_components));
    
    % Whitening
    W_white = D_reduced^(-1/2) * V_reduced';
    X_white = W_white * X_centered;
    
    
    W = zeros(n_components, n_components);
    
    for c = 1:n_components
    
        wp = randn(n_components, 1);
        wp = wp / norm(wp);
        
        for iter = 1:iterations
            wp_old = wp;
            
    
            % g(u) = tanh(alpha * u)
            % g'(u) = alpha * (1 - tanh^2(alpha * u))
            u = wp' * X_white;
            g_u = tanh(alpha * u);
            g_prime_u = alpha * (1 - g_u.^2);
            
            wp = (1/n) * (X_white * g_u') - (mean(g_prime_u) * wp);
            
            % Gram-Schmidt orthogonalization
            for j = 1:c-1
                wp = wp - (wp' * W(j,:)') * W(j,:)';
            end
            
            wp = wp / norm(wp);
            
            if abs(abs(wp' * wp_old) - 1) < epsilon
                fprintf('Component %d converged after %d iterations\n', c, iter);
                break;
            end
        end
        
        W(c, :) = wp';
    end
    
    % Independent components
    ica_result = W * X_white;
    
    % Mixing matrix
    A = W_white' * W';
    
    fprintf('Found %d independent components.\n', n_components);
end