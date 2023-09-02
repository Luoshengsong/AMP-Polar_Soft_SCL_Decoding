function V = polarEncoderSCL(U)
    %N x Ka
    % N = length(u)
    [N, Ka] = size(U);
    V = zeros(N, Ka);
    n = fix(log2(N));
    for k = 1: Ka
        v = U(:,k);
        for ll = 1:n
            N_iter = 2^ll;
            N_iter_half = N_iter/2;
            for kk = 1:(N/N_iter)
                range_start = (kk-1)*N_iter + 1;  % 起始索引
                range_mid = range_start + N_iter_half - 1;  % 中间索引
                range_end = kk*N_iter;  % 结束索引
                v(range_start:range_mid) = xor(v(range_start:range_mid), v(range_mid+1:range_end));
            end
        end
        V(:, k) = v;
    end
end