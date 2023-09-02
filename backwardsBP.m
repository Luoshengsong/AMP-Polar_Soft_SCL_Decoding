function [Left_Msg, Right_Msg] = backwardsBP(idx_frozen, llr, Msg_right_up, Msg_right_down)
    N = size(llr, 2);
    n = log2(N);
    n = floor(n);

    % transpose llr, namely Left_Msg
    Left_Msg = llr';
    % initialization of right message, namely, Right_Msg
    Right_Msg = zeros(size(Left_Msg)); % note that llr has been transposed
    Right_Msg(idx_frozen, 1) = inf;

    for layer = 1:n
        for i = 1:(N/2)
            % i'        <--> Msg_right_up(i, :)
            % i' + 2^j' <--> Msg_right_down(i, :)
            Right_Msg(Msg_right_up(i, layer), layer+1) = fFunction(Right_Msg(Msg_right_up(i, layer), layer), ...
                                                                    Right_Msg(Msg_right_down(i, layer), layer) + ...
                                                                    Left_Msg(Msg_right_down(i, layer), layer + 1));
            Right_Msg(Msg_right_down(i, layer), layer+1) = fFunction(Right_Msg(Msg_right_up(i, layer), layer), ...
                                                                     Left_Msg(Msg_right_up(i, layer), layer+1)) + ...
                                                           Right_Msg(Msg_right_down(i, layer), layer);
        end
    end
end

function c = fFunction( a,b)
    c = sign(a)*sign(b)*min(abs(a),abs(b));
end