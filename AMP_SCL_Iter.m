function Info_Bits_hat = AMP_SCL_Iter(Y, H_norm, beta, para)

constell = para.constell;
sigma2 = para.sigma2;
Ka = para.Ka;
outerIt = para.outerIt;
AMPIter = para.AMPIter;

Prob_constells = para.Prob_constells;  % N x Ka x length(constell)

Info_Bits_hat = nan( Ka, para.K0 + para.K_crc  );

% beta is a row vector
for it = 1: outerIt
    % AMP
    [X_effect_hat, X_effect_var] = AMP_detect(Y, H_norm, beta, constell, Prob_constells, sigma2, Ka, AMPIter);
    % Update X
    Xhat = diag(1./beta) * X_effect_hat ;
    Xvar = diag(1./ (beta.^2) ) * X_effect_var;

    % LLR conversion: BPSK
    LLR = ( - abs( Xhat + 1).^2 + abs( Xhat - 1).^2 ) ./ (2 * Xvar);
    LLR =  max(min(LLR, 40), -40);  % Ka x N

    % de-interleaving:
    deinlv_LLR = para.de_interlv_func(LLR);
    LLR_ext = zeros(Ka, para.N);
    for k = 1: Ka

%         PolarSCL = PolarSCL( deinlv_LLR(k,:)', para.if_information_bit, para.information_indices, para.list_sizeL, para.K0 + para.K_crc);
%         [PM, uhat_list, llrs_] = PolarSCLPolar_Decoder_SCL();

        LLR2converted= py.numpy.array(deinlv_LLR(k,:)');
    
        [SCL_output] = pyrunfile("Polar_Decoder_SCL.py", ...
                         "Returnlist", llr_channel = LLR2converted, ...
                                       if_information_bit_ = para.if_information_bit, ...
                                       information_indices = para.information_indices, ...
                                       L_= para.list_size, ...
                                       k_crc = para.K0 + para.K_crc);
        PM = SCL_output{1}; 
        PM = double(PM);
        uhat_list = SCL_output{2};
        uhat_list = double(uhat_list);
        LLR_list = SCL_output{3};
        LLR_list = cell(LLR_list);
        
        [~, sortIDs] = sort(PM, 'ascend');
        % most promising candidate
        promise_Id = 1;
        if para.CRCenable
            passCRC_flag = false;
            for i = 1: length(PM)
                % sortIDs(i)
                % check uhat_list()
                [blk, err] = para.CRC.decode(uhat_list(sortIDs(i), :)');
                if ~err
                    passCRC_flag = true;
                    u0_hat = blk;
                    promise_Id = i;
                    break;
                end
            end
            if passCRC_flag
                uhat = para.CRC.encode(u0_hat);
            else
                uhat = uhat_list(sortIDs(1), :);
            end
        else
            uhat = uhat_list(sortIDs(1), :);
        end
        
        % re-encode
        u_includesFrozen = zeros(para.N, 1);
        u_includesFrozen(para.idx_inf + 1, :) = uhat';
        encoded_bits_hat = polarEncoderSCL(u_includesFrozen);

        % BP backwards
        llr = double( LLR_list{sortIDs(promise_Id)} );
        [~, Right_Msg] = backwardsBP(para.idx_frozen+1, llr, para.Msg_right_up, para.Msg_right_down);
        
        % LLR flip
        llr_ext = Right_Msg(:,end);
        sign_distinct = sign(llr_ext) - (1 - 2 * encoded_bits_hat);
        llr_ext( sign_distinct~= 0 ) = -llr_ext( sign_distinct~= 0 );
        LLR_ext(k, :) = llr_ext';

        if it == outerIt
            Info_Bits_hat(k, :) = uhat;
        end
    end

    % interleaving
    LLR_ext = para.interlv_func(LLR_ext);

    % convert LLR into soft-symbol message
    Prob_bit0 = exp(LLR_ext) ./ (1 + exp(LLR_ext));
    Prob_bit1 = 1 - Prob_bit0;
    Prob_constells(:,:,1) = Prob_bit1';
    Prob_constells(:,:,2) = Prob_bit0';


end

% Info_Bits_hat = Info_Bits_hat';

end