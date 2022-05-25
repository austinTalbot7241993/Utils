'''

'''
import numpy as np
from scipy.special import loggamma,digamma

def dirichlet_kl_divergence_np(alpha,beta):
    '''

    '''
    alpha0 = np.sum(alpha)
    beta0 = np.sum(beta)
    t1 = loggamma(alpha0)
    t2 = -1*np.sum(loggamma(alpha))
    t3 = -1*loggamma(beta0)
    t4 = np.sum(loggamma(beta))
    t5 = np.sum((alpha-beta)*(digamma(alpha)-digamma(np.sum(alpha))))
    return t1 + t2 + t3 + t4 + t5

def dirichlet_entropy_np(alpha):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    K = len(alpha)
    alpha_0 = np.sum(alpha)

    log_B_alpha_num = np.sum([loggamma(alpha[i]) for i in range(K)])
    log_B_alpha_denom = loggamma(alpha_0)

    term1 = log_B_alpha_num - log_B_alpha_denom
    term2 = (alpha_0-K)*digamma(alpha_0)
    term3 = np.sum([(alphaj-1)*digamma(alphaj) for alphaj in alpha])

    entropy = term1 + term2 + term3
    return entropy




