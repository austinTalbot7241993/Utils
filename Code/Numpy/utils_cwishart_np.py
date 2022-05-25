'''
This implements methods related to the complex Wishart and inverse Wishart
distributions

Methods:


Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import numpy.linalg as la
from scipy.special import gammaln

def cmgammaln_np(nu,p):
    '''

    '''
    out = 0.5*p*(p-1)*np.log(np.pi)
    for i in range(p):
        out += gammaln(nu-i)
    return out
    

def ciwish_loglikelihood_np(X,nu,Psi):
    '''

    lp(X|nu,Psi)=nu log|Psi|- log CG(nu) -(nu+p)|X|-tr(PsiX^{-1})

    Parameters
    ----------

    Returns
    -------

    '''
    p = Psi.shape[0]
    _,ldP = la.slogdet(Psi)
    _,ldX = la.slogdet(X)
    term1 = nu*ldP - cmgammaln_np(nu,p)
    term2 = -(nu+p)*ldX
    term3 = -np.trace(la.solve(X,Psi))
    return term1 + term2 + term3

def cwish_loglikelihood_np(X,nu,Psi):
    '''

    lp(X|nu,Psi)=(nu-p)log|X| - tr(Psi^{-1}X) - p log|Psi|- log CG(nu)

    Parameters
    ----------

    Returns
    -------

    '''
    p = Psi.shape[0]
    _,ldP = la.slogdet(Psi)
    _,ldX = la.slogdet(X)
    term1 = -p*ldP - cmgammaln_np(nu,p)
    term2 = (nu-p)*ldX
    term3 = -np.trace(la.solve(Psi,X))
    return term1 + term2 + term3

def ciwish_sample_np(nu,Sigma,nsamples=1):
    pass

def cwish_sample_np(nu,Sigma,nsamples=1):
    pass


