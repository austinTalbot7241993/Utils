import numpy as np
from scipy.fft import fft,ifft

def positive_fft_3d(X):
    '''
    Paramters
    ---------
    X : array-like (N,p,nT)
        

    Returns
    -------

    '''
    N,p,nT = X.shape
    print(N,p,nT)
    Nf = int(np.floor(nT/2))
    out = np.zeros((N,p,Nf)).astype(np.complex64)
    for i in range(N):
        for j in range(p):
            fft_sub = fft(X[i,j,:]).astype(np.complex64)
            out[i,j,:] = fft_sub[1:Nf+1]
    return out


