'''
This implements methods related to the complex gaussian distribution in 
numpy

Methods:


Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import numpy.linalg as la

def cmvn_loglikelihood_np(X,Sigma):
    n = Sigma.shape[0]
    term1 = -n*np.log(np.pi)
    _,nterm2 = la.slogdet(Sigma)
    term3 = 

def cmvn_sample_np(Sigma,n_samples=1):
    


