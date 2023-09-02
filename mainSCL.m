clc; clear all;
rng('default');
warning('off');

poolobj = gcp('nocreate'); delete(poolobj);
% p = parpool(16);

%% Setting 
CRCenable = true;
list_size = 16;
SNRdB = 4 : 2: 12;%[0, 4, 8, 12, 16];
monteCarlos = [1e2, 1e2, 1e2, 1e3, 1e3, 1e3, 1e3, 1e4, 1e4];
para.constell = [1, -1];  % BPSK modulation
para.CRCenable = CRCenable;
para.outerIt = 6;
para.AMPIter = 15;

para.Ka = 16; % number of active users
M = 32;  % number of antennas
pathloss = ones(para.Ka, 1);

K0 = 24;
K_crc = 8;
if ~CRCenable
    K_crc = 0;
end
N = 64;
% Rate = (K0 + K_crc) / N;
para.K0 = K0;
para.N = N;
para.K_crc = K_crc;
para.CRCenable = CRCenable;

%% Preporcessing
% interleaver and de-interleaver related: for row vector
interleaver_IDs = randperm(N);
[~, de_interleaver_IDs] = sort(interleaver_IDs);
interlv_func = @(Bits)  Bits(:, interleaver_IDs);  % Bits could be a matrix with size (No_Active_Users x FEC_Len)
de_interlv_func = @(interlv_Bits) interlv_Bits(:, de_interleaver_IDs);
para.interlv_func = interlv_func;
para.de_interlv_func = de_interlv_func;

if CRCenable
    CRC = CRC(K_crc);
    para.CRC = CRC;
end

% Preprocessing (Regarding **Soft List Decoding**)
[Prepro] = pyrunfile("PreprocessCode.py","Returnlist", N_ = N, k_ = K0 + K_crc);
idx_frozen = Prepro{1};
idx_frozen = double(idx_frozen);
idx_inf = Prepro{2};
idx_inf = double(idx_inf);
if_information_bit = Prepro{3};

[Msg_right_up, Msg_right_down] = index_Matrix(N);

para.idx_frozen = idx_frozen;
para.if_information_bit = if_information_bit;
para.information_indices = Prepro{2};
para.idx_inf = idx_inf;
para.list_size = list_size;
para.Msg_right_up = Msg_right_up;
para.Msg_right_down = Msg_right_down;

%% executation
% initialization
para.Prob_constells = ones(N, para.Ka, length(para.constell)) ./ length(para.constell);

BER = nan(length(SNRdB), 1);
FER = nan(length(SNRdB), 1);
for snr_id = 1: length(SNRdB)
    sigma2 = N / 10^(SNRdB(snr_id) / 10);
    para.sigma2 = sigma2;
    disp(['SNR = ' , num2str( SNRdB(snr_id) ) , 'dB.[', num2str(snr_id) ' of ', num2str(length(SNRdB)), ']']);
    tic,
    BER_SNR = nan(monteCarlos(snr_id), 1);
    FER_SNR = nan(monteCarlos(snr_id), 1);

    for mt = 1: monteCarlos(snr_id)

        % --------------------- Transmitter side --------------------------
        % info bits
        u0 = randi([0, 1], K0, para.Ka);
        % append CRC
        if CRCenable
            u = CRC.encode(u0); % (K0 + K_crc) x Ka
        else
            u = u0;
        end
        % size u: (K0 + k_crc) x Ka
        u_includesFrozen = zeros(N, para.Ka);
        u_includesFrozen(idx_inf + 1, :) = u;

        % Polar encoding
        % encodedBits = Polar_SCAN.Polar_SCAN_encode(u'); % Ka x N
        encoded_bits = polarEncoderSCL(u_includesFrozen);

        % interleaving
        inlv_bits = interlv_func(encoded_bits');
        % BPSK modulation
        X = 2 * inlv_bits - 1;

        %---------------- Channel effect (BPSK, only I-path) --------------
        H = sqrt(1) * ( randn(M, para.Ka) );
        beta = sqrt( sum(abs(H).^2, 1) );
        H_norm = H ./ beta;
        
        % ---------------------- Receiver side ----------------------------
        Y = H * X + sqrt(1 * sigma2) * ( randn(M, N) );
        % perform AMP detect: output the likelihood information
        Info_Bits_hat = AMP_SCL_Iter(Y, H_norm, beta, para);

        % BER
        bit_comp = bitxor(Info_Bits_hat, u');
        BER_SNR(mt) = sum(bit_comp(:)) / para.Ka / N;
        
        % FER
%         [~, idx_check] = ismember(Info_Bits_hat, u', 'rows');
%         idx_check = idx_check(idx_check > 0);
        checkFER = sum(bit_comp, 2);
        FER_SNR(mt) = sum(checkFER > 0)  / para.Ka;
       
    end
    BER(snr_id) = sum(BER_SNR) / monteCarlos(snr_id),
    FER(snr_id) = sum(FER_SNR) / monteCarlos(snr_id),
    toc,
    disp('--------------------------------------');
end

% delete(p);

figure; semilogy( SNRdB, BER, 'm-o' ,...
              SNRdB, FER, 'b-s');
xlabel('SNR(dB)');
grid on;
legend('BER', 'FER')