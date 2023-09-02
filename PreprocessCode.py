import numpy as np
import polarLib as polar

# require to input N_ and k_
design_snr = -2
n_ = np.log2(N_)
n_ = int(n_)
N_ = int(N_)
k_ = int(k_)
# k: # number of information bits + CRC bits
# Find bit-channel reliabilities using the Bhattacharyya parameter
Z   = polar.constructPolarCode(N_, design_param = design_snr)
idx_inf_plus_frozen = polar.bitReverseArray(np.arange(N_), n_)   # array([0, 4, 2, 6, 1, 5, 3, 7])
Z_inf_frozen        = Z[idx_inf_plus_frozen]        
argsort_list        = np.argsort(Z_inf_frozen)
idx_used            = idx_inf_plus_frozen[argsort_list]
idx_shortened       = polar.bitReverseArray(np.arange(N_, N_), n_)
idx_frozen          = np.copy(idx_used[k_:])
idx_inf             = np.copy(idx_used[:k_])

if_information_bit  = np.zeros(N_)       # "0" denotes that this bit is frozen
if_information_bit[idx_inf]        = 1   # "1" denotes that this bit is information
# 0: idx_inf = [7, 6, 5, 3] corrresponds information_bit index 
# 1: idx_frozen = [4, 2, 1, 0] corresponds frozen_bit index
# 2: in such case, wo do not consider shortened!
if_information_bit[idx_shortened]  = 2   # "2" denotes that this bit is shortened

Returnlist = [idx_frozen, idx_inf, if_information_bit]