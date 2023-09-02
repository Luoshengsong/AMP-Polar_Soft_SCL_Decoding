# AMP-Polar_Soft_SCL_Decoding

**Note**: This procedure is implemented on MATLAB, where a few Python functions are attached. And these Python functions are called by the "pyrunfile" of MATLAB. Readers can refer to [the website](https://aleksandarhaber.com/tutorial-on-how-to-execute-python-code-directly-matlab/) for more details.

We consider an uplink wireless channel MIMO model where there are 

- $M$ antennas at BS
- $K$ single antenna terminals

Thus, we denote the effective channel as $\boldsymbol H \in \mathbb C^{M \times K}$, which is Rayleigh distributed. This system model is characterized as 
$$\boldsymbol Y = \boldsymbol H \boldsymbol X + \boldsymbol W$$
where the element of $\boldsymbol X$ is BPSK modulated, and the element of $\boldsymbol W$ is AWGN.

Assume perfect CSI is known in BS side, we construct an iterative Bayesian receiver based on Turbo structure:

[**Module-1**: MIMO Detector] <-- $\Pi$ --> [**Module-2**: Polar Soft-SCL decoding]

The simulation result:

![image](https://github.com/Luoshengsong/AMP-Polar_Soft_SCL_Decoding/assets/73685146/8ed3dab7-ff1f-4846-ad93-34520d7bc35b)

![image](https://github.com/Luoshengsong/AMP-Polar_Soft_SCL_Decoding/assets/73685146/d9dc0ff6-4408-4592-a59e-01abf14210b8)

![image](https://github.com/Luoshengsong/AMP-Polar_Soft_SCL_Decoding/assets/73685146/99832d6e-b55e-42da-9645-112cdd00045e)




## Reference
[1] A. Balatsoukas-Stimming, M. B. Parizi and A. Burg, "LLR-Based Successive Cancellation List Decoding of Polar Codes," 2015 (TSP).

[2] L. Xiang, Y. Liu, Z. B. K. Egilmez, R. G. Maunder, L. -L. Yang and L.Hanzo, "Soft List Decoding of Polar Codes," 2020 (TVT)


