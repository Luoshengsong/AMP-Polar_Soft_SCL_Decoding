function [Xhat, Xvar] = AMP_detect(Y, H_norm, beta, constell, Prob_constells, sigma2, Ka, maxIter)
    [M, N] = size(Y);
    damp = 0.99;
    Xhat = zeros(Ka, N);
    Xvar = zeros(Ka, N);
    for n = 1: N
        % the n-th vector of X
        Prob_constell = squeeze( Prob_constells(n,:,:) );
        y = Y(:, n);
        r = y;
        tau2 = norm(r)^2 / M;
        x = zeros(Ka, 1);
        
        for it = 1: maxIter
            xold = x;
            derivatives = zeros(Ka, 1);
            for k = 1: Ka
                Prob_constell_k = Prob_constell(k, :);
                xk_hat = transpose(r) * conj( H_norm(:, k) ) + xold(k);
                [xk, derivative] = denoiser(xk_hat, beta(k), tau2, constell, Prob_constell_k, sigma2);
                x(k) = xk;
                derivatives(k) = derivative;
            end
            r = y - H_norm * x + Ka/M * mean(derivatives) * r;
            tau2 = norm(r)^2 / M;
            %mse(it) = norm(x - x_true)^2 / M;
            x = damp * x + (1 - damp) * xold;
        end
        % likelihood CN(Xhat, Xvar)
        Xhat(:, n) = x;
        Xvar(:, n) = tau2;
    end
end

function [xk, derivative] = denoiser(xk_hat, beta_k, tau2, constell, Prob_constell, sigma2)
    coes = zeros(length(constell), 1);
    for i = 1: length(constell)
        if isreal( constell(2) )
            log_interior_value = - abs(xk_hat - beta_k * constell(i) )^2 / (2 * tau2);
        else
            log_interior_value = - abs(xk_hat - beta_k * constell(i) )^2 / (tau2);
        end
        coes(i) = Prob_constell(i) * exp( log_interior_value );
    end
    xk = dot(coes, beta_k * transpose(constell) ) / sum(coes);
    second_order_moment = dot(coes, abs( beta_k * transpose(constell) ).^2 ) / sum(coes);
    variance = second_order_moment - abs(xk)^2;
    derivative = variance / tau2;
    derivative = max(derivative, 1e-6/sigma2); % numerical robusteness 
end