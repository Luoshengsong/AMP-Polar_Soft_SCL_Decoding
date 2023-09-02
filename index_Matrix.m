function [Msg_right_up, Msg_right_down] = index_Matrix(N)
    n = log2(N);
    n = floor(n);
    Msg_right_up = zeros(N/2, n);
    Msg_right_down = zeros(N/2, n);
    for layer = 1:n
        selected = [];
        cnt = 1;
        for i = 1:N
            if ismember(i, selected)
                continue;
            else
                Msg_right_up(cnt, layer) = i;
                Msg_right_down(cnt, layer) = i + 2^(layer-1);
                selected = [selected, i, i + 2^(layer-1)];
                cnt = cnt + 1;
            end
        end
    end
end


