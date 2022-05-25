'''
This implements methods related to the gaussian distribution in numpy

Methods:


mvn_kl_divergence_np

Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import numpy.linalg as la
from utils_matrix_np import evaluate_invquad_vec_np
from utils_matrix_np import evaluate_quad_np

#################################
##                             ##
## Information theory section  ## 
##                             ##
#################################

def mvn_kl_batch_standard_np(mu,Sigma):
    pass

def mvn_kl_divergence_np(Sigma1,Sigma2,mu1=None,mu2=None):
    '''
    Computes the kl divergence between two gaussian distributions.

    p ~ N(mu1,S1)
    q ~ N(mu2,S2)

    KL(p|q)

    Parameters
    ----------
    Sigma1 : np.array-like(p,p)
        First covariance matrix

    Sigma2 : np.array-like(p,p)
        Second covariance matrix

    mu1 : np.array-like(p,)
        First mean (optional)

    mu2 : np.array-like(p,)
        Second mean (optional)

    Returns
    -------
    kl_divergence : float
        Divergence between two distributions
    '''
    p = Sigma1.shape[0]
    _,logdet2 = la.slogdet(Sigma2)
    _,logdet1 = la.slogdet(Sigma1)
    term1 = logdet2 - logdet1
    term2 = -p
    term3 = np.trace(la.solve(Sigma2,Sigma1))
    kl_divergence = term1 + term2 + term3 
    if mu1 is not None: 
        term4 = evaluate_invquad_vec_np(diff,Sigma2)
        kl_divergence += term4
    kl_divergence *= 0.5
    return kl_divergence

def mvn_kl_divergence_eye_np(Sigma,mu=None):
    '''
    Computes the kl divergence between a distribution and standard normal

    p ~ N(mu1,S1)
    q ~ N(0,I)
    KL(p|q)

    Parameters
    ----------
    Sigma : np.array-like(p,p)
        Covariance matrix

    mu1 : np.array-like(p,)
        Mean (optional)

    Returns
    -------
    kl_divergence : float
        Divergence between two distributions
    '''
    p = Sigma.shape[0]
    _,logdet = la.slogdet(Sigma)
    term1 = logdet
    term2 = -p
    term3 = np.trace(la.inv(Sigma))
    kl_divergence = term1 + term2 + term3 
    if mu is not None: 
        term4 = evaluate_invquad_vec_np(mu,Sigma)
        kl_divergence += term4
    kl_divergence *= 0.5
    return kl_divergence

def mvn_conditional_distribution_np(Sigma,idx_obs):
    '''
    Computes the distribution parameters p(X_miss|X_obs)

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        The covariance matrix

    idx_obs: np.array-like,(p,)
        The observed covariates

    Returns
    -------
    mu_bar : array-like
        The conditional mean

    Sigma_bar : array-like
        Conditional covariance
    '''

    Sigma_sub = Sigma[idx_obs==1]
    Sigma_22 = Sigma_sub[:,idx_obs==1]
    Sigma_21 = Sigma_sub[:,idx_obs==0]
    Sigma_nonsub = Sigma[idx_obs==0]
    Sigma_11 = Sigma_nonsub[:,idx_obs==0]
    mu_bar = la.solve(Sigma_22,Sigma_21)
    Second_part = np.dot(Sigma_21.T,mu_bar)

    Sigma_bar = Sigma_11 - Second_part
    return mu_bar.T,Sigma_bar

def mvn_entropy_np(Sigma):
    '''
    Computes the entropy of a multivariate Gaussian distribution

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        The covariance matrix

    Returns
    -------
    entropy : float
        The entropy (in nats)
    '''
    k = cov.shape[1]
    cov_new = 2*np.pi*np.e*cov
    _,logdet = la.slogdet(cov_new)
    entropy = 0.5*logdet
    return entropy

def mvn_mutual_information_np(Sigma,idxs):
    '''
    Returns the mutual information between x_{idxs==1} and x_{idxs==0} 
    for data that comes from a multivariate normal distribution

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        The covariance matrix

    idxs : np.array-like,(p,) boolean
        The different

    Returns
    -------

    '''
    Hy = mvn_entropy(Sigma[np.ix_(idxs,idxs)])
    Prod,Sigma_bar = mvn_conditional_parameters_np(Sigma,idxs)
    Hyx = mvn_entropy(Sigma_bar)
    return Hy - Hyx

####################################
##                                ##
## Statistics/likelihood section  ## 
##                                ##
####################################

def mvn_loglikelihood_np(X,Sigma):
    '''
    Evaluates the average loglikelihood of samples X with distribution 
    parameterized by Sigma

    Parameters
    ----------
    X : array-like,(N,k)
        Data

    Sigma : np array-like=(k,k)
        The covariance matrix

    Returns
    -------
    log_likelihood
        The average log likelihood
    '''
    if X.ndim == 1:
        X = np.atleast_2d(X).T
    if (type(Sigma) == type(1.0)) or (Sigma.ndim == 0): 
        Sigma = Sigma*np.ones((1,1))
    k = Sigma.shape[0]
    term1 = -k/2*np.log(2*np.pi)
    _,logdet = la.slogdet(Sigma)
    term2 = -1/2*logdet
    quad_init = la.solve(Sigma,np.transpose(X))
    diff_big = X*np.transpose(quad_init)
    row_sum = np.sum(diff_big,axis=1)
    quad = np.mean(row_sum)
    term3 = -1/2*quad
    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_conditional_likelihood_np(X,Sigma,idx_obs):
    '''
    Evaluates the conditional likelihood of samples conditioned on observed
    covariates

    Parameters
    ----------
    X : array-like,(N,k)
        Data

    Sigma : np array-like=(k,k)
        The covariance matrix

    idx_obs : np boolean array,(k,)
        The observed covariates 

    Returns
    -------
    log_likelihood
        The average log likelihood
    '''
    X_obs = X[:,idx_obs]
    X_miss = X[:,idx_obs==False]
    coef_,sig_ = mvn_conditional_distribution_np(Sigma,idx_obs)
    mu_ = np.dot(X_obs,coef_.T)
    log_likelihood = mvn_loglikelihood_np(X[:,idx_obs==False]-mu_,sig_)
    return log_likelihood


def mvn_quadratic_form_np(Lambda,Sigma,Mu):
    '''
    https://en.wikipedia.org/wiki/Quadratic_form_(statistics)

    '''
    if len(Mu.shape) == 1:
        mlm = Mu.dot(np.dot(Lambda,Mu.T))
    else:
        mlm = evaluate_quad_np(Mu,Lambda)
    try:
        tLS = np.trace(np.dot(Lambda,Sigma))
    except:
        tLS = Lambda*Sigma
    return mlm + tLS


        
























    

