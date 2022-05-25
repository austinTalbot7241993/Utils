'''
This implements methods relevant to Gaussian distributions in Tensorflow 

Methods:

mvn_loglikelihood_tf(X,Sigma)
    Evaluates the average loglikelihood of samples X with distribution 
    parameterized by Sigma

mvn_loglikelihood_mean_tf(X,mu,Sigma)
    Evaluates the average loglikelihood of samples X with distribution 
    parameterized by Sigma and mu

mvn_conditional_PPCA_tf(W_,sigma_,idxs)
    Computes the conditional distribution when the distribution is 
    N(0,W_W_^T+sigma_I)

mvn_sample_tf(mu,Sigma,lamb=0.0)
    This samples from a normal distribution with potential tikhonov 
    regularization to ensure positive definiteness

mvn_conditional_tf(Sigma,idx_obs)
    Computes the distribution parameters p(X_miss|X_obs) for a general
    distribution N(0,Sigma)

mvn_conditional_mean_tf(mu,Sigma,idx_obs)

Author : Austin Talbot <austin.talbot1993@gmail.com>

'''

import numpy as np
import tensorflow as tf
from utils_matrix_tf import evaluate_invquad_mat_tf,evaluate_invquad_vec_tf

def mvn_kl_batch_diag_tf(mu,logvar):
    '''
    Uses mean parameterization and log-variance, as commonly used in VAEs.

    Parameters
    ----------

    Returns
    -------

    '''
    kl_d1_t = -0.5*tf.reduce_sum(1.0+logvar-tf.square(mu)-
                        tf.exp(logvar),axis=1)
    kl_val = tf.reduce_mean(kl_d1_t)
    return kl_val

def mvn_kl_batch_standard_tf(mu,Sigma):
    '''


    Parameters
    ----------
    mu : 

    Sigma : 

    Returns
    -------

    '''
    p = mu.shape[1]
    logdet = tf.linalg.logdet(Sigma)
    term1 = -logdet
    term2 = -p
    term3 = tf.linalg.trace(Sigma)
    term4 = tf.reduce_mean(tf.reduce_sum(tf.square(mu),axis=1))
    kl_divergence = 0.5*(term1 + term2 + term3 + term4)
    return kl_divergence

def mvn_kl_divergence_tf(Sigma1,Sigma2,mu1=None,mu2=None):
    '''
    kl(Sig2|Sig1)


    Parameters
    ----------
    mu : 

    Sigma : 

    Returns
    -------

    '''
    logdet2 = tf.linalg.logdet(Sigma2)
    logdet1 = tf.linalg.logdet(Sigma1)
    prod = tf.linalg.solve(Sigma2,Sigma1)
    term1 = logdet2 - logdet1
    term2 = -p
    term3 = tf.linalg.trace(prod)
    kl_divergence = term1 + term2 + term3
    if mu1 is not None:
        diff = mu1 - mu2
        kl_divergence += evaluate_invquad_vec_tf(diff,Sigma2)
    return 0.5*kl_divergence

def mvn_loglikelihood_tf(X,Sigma):
    '''
    Evaluates the average loglikelihood of samples X with distribution 
    parameterized by Sigma

    Parameters
    ----------
    X : array-like,(N,k)
        Data

    Sigma : tf array-like=(k,k)
        The covariance matrix

    Returns
    -------
    log_likelihood
        The average log likelihood
    '''
    k = Sigma.shape[0]
    term1 = -k/2*tf.math.log(2*np.pi)
    term2 = -1/2*tf.linalg.logdet(Sigma)
    quad_init = tf.linalg.solve(Sigma,tf.transpose(X))
    diff_big = tf.multiply(X,tf.transpose(quad_init))
    row_sum = tf.reduce_sum(diff_big,axis=1)
    quad = tf.reduce_mean(row_sum)
    term3 = -1/2*quad
    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_loglikelihood_noinv_tf(X,Sigma,Sigma_inv):
    '''

    '''
    k = Sigma.shape[0] #1
    term1 = -k/2*tf.math.log(2*np.pi) # 1
    term2 = -1/2*tf.linalg.logdet(Sigma) # 1
    quad_init = tf.matmul(X,Sigma_inv) # N x p
    diff_big = tf.multiply(X,quad_init) # N x p
    row_sum = tf.reduce_sum(diff_big,axis=1) # N
    quad = tf.reduce_mean(row_sum) # 1
    term3 = -1/2*quad # 1
    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_loglikelihood_iso_tf(X,sigma2):
    '''
    Evaluates the loglikelihood of samples X with distribution 
    N(0,sigma^2I). Not necessary but gives computational gains

    Parameters
    ----------
    X : array-like,(N,k)
        Data

    sigma2 : float
        Isotropic noise

    Returns
    -------
    log_likelihood
        The average log likelihood
    '''
    k = X.shape[1]
    term1 = -k/2*tf.math.log(2*np.pi)
    term2 = -k/2*tf.math.log(sigma2)
    row_sum = tf.reduce_sum(tf.square(X),axis=1)
    term3 = -1/(2*sigma2)*tf.reduce_mean(row_sum)
    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_loglikelihood_mean_tf(X,mu,Sigma):
    '''
    Evaluates the average loglikelihood of samples X with distribution 
    parameterized by Sigma and mu

    Parameters
    ----------
    X : array-like,(N,k)
        Data

    mu : array-like,(N,k) or (k,)
        The mean

    Sigma : tf array-like=(k,k)
        The covariance matrix

    Returns
    -------
    log_likelihood
        The average log likelihood

    '''
    diff = X - mu
    return mvn_loglikelihood_tf(diff,Sigma)

def mvn_sample_tf(mu,Sigma,lamb=0.0):
    '''
    This samples from a normal distribution with potential tikhonov 
    regularization to ensure positive definiteness

    Parameters
    ----------
    mu : tf.array,(?,p)
        The mean

    Sigma : tf.array,(?,p,p)
        The covariance matrix

    lamb : float,default=0.0
        The tikhonov regularization

    Returns
    -------
    z_sample : tf.Array,(?,p)
        The sampled values
    '''
    if lamb == 0.0:
        chol = tf.linalg.cholesky(Sigma)
    else:
        p = Sigma.shape[-1]
        eye = tf.constant(np.eye(p).astype(np.float32))
        chol = tf.linalg.cholesky(Sigma + eye)

    eps = tf.random.normal(shape=mu.shape)
    if len(chol.shape) == 2:
        z_sample = tf.matmul(eps,tf.transpose(chol)) + mu
    else:
        eps_e = tf.expand_dims(eps,axis=-1)
        mul = tf.multiply(eps_e,tf.transpose(chol,perm=(0,2,1)))
        prod = tf.reduce_sum(mul,axis=1)
        z_sample = prod + mu

    return z_sample

def mvn_conditional_tf(Sigma,idx_obs):
    ''' 
    Computes the distribution parameters p(X_miss|X_obs) for a general
    distribution N(0,Sigma)

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
    cov_sub_1 = tf.boolean_mask(Sigma,idx_obs==1,axis=0)
    cov_sub_0 = tf.boolean_mask(Sigma,idx_obs==0,axis=0)
    cov_11 = tf.boolean_mask(cov_sub_1,idx_obs==1,axis=1)
    cov_10 = tf.boolean_mask(cov_sub_1,idx_obs==0,axis=1)
    cov_00 = tf.boolean_mask(cov_sub_0,idx_obs==0,axis=1)
    
    Sigma_init = tf.linalg.solve(cov_11,cov_10)
    mu_bar = tf.transpose(Sigma_init)

    quad = tf.matmul(tf.transpose(cov_10),Sigma_init)
    Sigma_bar = cov_00 - quad

    return mu_bar,Sigma_bar

def mvn_conditional_mean_tf(mu,Sigma,idx_obs):
    '''
    Computes the distribution parameters p(X_miss|X_obs) for a general
    distribution N(mu,Sigma)

    Parameters
    ----------
    mu : tf.array-like,(p,)
        The mean

    Sigma : tf.array-like,(p,p)
        The covariance matrix

    idx_obs: tf.array-like,(p,)
        The observed covariates

    Returns
    -------
    mu_bar : array-like
        The conditional mean

    Sigma_bar : array-like
        Conditional covariance
    '''
    mu_1 = tf.boolean_mask(mu,idxs==1)
    mu_0 = tf.boolean_mask(mu,idxs==0)
    cov_sub_1 = tf.boolean_mask(cov,idxs==1,axis=0)
    cov_sub_0 = tf.boolean_mask(cov,idxs==1,axis=0)
    cov_11 = tf.boolean_mask(cov_sub_1,idxs==1,axis=1)
    cov_10 = tf.boolean_mask(cov_sub_1,idxs==0,axis=1)
    cov_00 = tf.boolean_mask(cov_sub_0,idxs==0,axis=1)

    Sigma_init = tf.linalg.solve(cov_11,cov_10)

    quad = tf.matmul(tf.transpose(cov_10),Sigma_init)
    Var = cov_00 - quad

    return mu_1,mu_0,tf.transpose(Sigma_init),Var

#Mu_p is allowed to be a matrix
def kl_divergence_mvg_batch(mu_p,mu_q,cov_p,cov_q):
    logdet_p = tf.linalg.logdet(cov_p)
    logdet_q = tf.linalg.logdet(cov_q)
    prod = tf.linalg.solve(cov_q,cov_p)
    d = cov_p.shape[0]
    N = mu_p.shape[0]

    diff = mu_p - mu_q
    end = tf.linalg.solve(cov_q,tf.transpose(diff))#N x p
    diff_e = tf.expand_dims(diff,axis=1)#N x 1 x p
    end_e = tf.expand_dims(end,axis=1)#p x 1 x N
    end_prod = tf.transpose(end_e,perm=(2,0,1)) #N x p x 1
    mmul = tf.matmul(diff_e,end_prod) # N x 1 x 1

    term_1 = logdet_q
    term_2 = logdet_p
    term_3 = tf.linalg.trace(prod)
    term_4 = tf.reduce_mean(mmul)

    KL_tot = 0.5*(term_1 - term_2 + term_3 + term_4 - d)
    return KL_tot


def quadratic_form_tf(Lambda,mu,Sigma):
    pass


def quadratic_form_batch_tf(Lambda,Mu,Sigma):
    pass
    
