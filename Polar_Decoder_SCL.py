import numpy as np
import polarLib as polar

def SCLDecoder(llr_channel, if_information_bit, information_indices, L, k_crc):
    '''
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
        **llr_channel** : array_like
            channel LLR values `numpy.float32`
        if_information_bit: array_like
            stores if the bits correspoding to the indices are information bits or not, type  `bool`
        L: list size of the SCL `integer`

    '''
    # Define Polar code length and depth
    N     = len(llr_channel)
    n     = int(np.log2(N))
    L = int(L)
    k_crc = int(k_crc)

    #malloc and initialize Decoder LLRS
    llrs  = -np.inf*np.ones((n+1,1<<n),dtype=np.float32)

    llrs[-1,:] = llr_channel
    llrs = [llrs]*L

    #HARD Decisions of SCL decoder
    s = -1*np.ones((n+1,int(2**n)),dtype=np.int8)
    s = [s]*L

    DM    = np.zeros((L))                        # path metrics
    PM    = np.inf*np.ones((L),dtype=np.float32) # cumulative path metrics
    PM[0] = 0.0
    PM_DM = np.zeros((2*L))


    #--- START OF SCL Decoder ---
    for ii in range(N):
        #---START OF "IF INFORMATION BIT"---
        if if_information_bit[ii] == 0:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = polar.Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
                PM[dd]        += -llrs[dd][0,ii]*(llrs[dd][0,ii]<0)
        #---END OF "IF INFORMATION BIT"---


        elif if_information_bit[ii] == 2:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = polar.Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
        #---END OF "IF INFORMATION BIT"---

        #---START OF "IF FROZEN BIT"---
        else: #information bit
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = polar.Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = llrs[dd][0,ii]<0
                DM[dd]         = np.abs(llrs[dd][0,ii])
        #---END OF "IF FROZEN BIT"---

        #SELECTING THE BEST "L" PATHS
        if if_information_bit[ii] and L>1:
            PM_DM[:L] = PM        #Path metrics
            PM_DM[L:] = PM + DM   #Path metric + Decision metrics
            idx_sort   = np.argsort(PM_DM) #sort path metrics

            #find decoders in the list need to be updated
            idx_min_low  = idx_sort[:L][idx_sort[:L]>=L]-L
            idx_min_up   = idx_sort[L:][idx_sort[L:]<L]

            # If decoders in the list need to be updated
            len_list_change = len(idx_min_low) # or len(idx_min_up)
            if  len_list_change != 0:

                #---START OF DECODER COPYING---
                for bb in range(len_list_change):
                    llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])  # LLR degerlerini tasi
                    s[idx_min_up[bb]]    = np.copy(s[idx_min_low[bb]])     # bitleri tasi
                    s[idx_min_up[bb]][0,ii] = 1-s[idx_min_low[bb]][0,ii]   # cozulen son biti flip et
                #---END OF DECODER COPYING---

                #Path metric guncelle
                PM[idx_min_up] = PM_DM[idx_min_low + L]
        #---END OF LIST UPDATE---
    #---END OF SCL DECODING---

    #Select the possible codeword having the minimum path metric
    dd_best = np.argmin(PM)
    # return codewords s of all paths
    uhat_list = np.zeros(( L, k_crc ), dtype=np.int8)
    #print(PM)
    for ii in range(L):
        uhat_list[ii, :] = s[ii][0,:][information_indices[:k_crc]]
    # return s[dd_best][0,:]
    return PM, uhat_list, llrs

PM, uhat_list, llrs = SCLDecoder(llr_channel, if_information_bit_, information_indices, L_, k_crc)
Returnlist = [PM, uhat_list, llrs]