'''
This implements methods relevant to Gaussian models specific to 
probabilistic PCA and its minor extensions. Remaining models will be left
alone in the previous file.


Methods:


Author : Austin Talbot <austin.talbot1993@gmail.com>

'''

import numpy as np
import tensorflow as tf
from utils_gaussian_tf import mvn_conditional_tf

pi2 = tf.constant(tf.math.log(2*np.pi))
oh = tf.constant(-1/2.)

def mvn_loglikelihood_isoLinear_tf(X,W,sigma2,eye=None,eye2=None):
    '''
    This computes the marginal likelihood of X under model

    p(X) = N(0,WW^T + \sigma^2I)

    Uses woodbury matrix identity to make very quick

    Parameters
    ----------

    Returns
    -------
    log_likelihood : tf.Float
        The average log likelihood 
    '''
    p,L = W.shape
    if eye is None:
        eye = tf.constant(np.eye(L).astype(np.float32))
    if eye2 is None:
        eye2 = tf.constant(np.eye(p).astype(np.float32))

    WT = tf.transpose(W)
    sIWTW = sigma2*eye + tf.matmul(WT,W) 

    logDet = tf.linalg.logdet(sIWTW/sigma2) + p*tf.math.log(sigma2)

    back = tf.linalg.solve(sIWTW,WT)
    prod_end = tf.matmul(W,back)
    C_inv = (eye2 - prod_end)/sigma2
    cov = tf.matmul(W,WT) + sigma2*eye2

    end = tf.matmul(X,C_inv)
    prod = tf.multiply(X,end)
    sums = tf.reduce_sum(prod,axis=1)

    term1 = -p/2*tf.math.log(2*np.pi)
    term2 = -1/2*logDet
    term3 = -1/2*tf.reduce_mean(sums)
    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_loglikelihood_diagLinear_tf(X,W,D,eye=None,eye2=None):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    pass

def mvn_conditional_params_isoLinear_tf(W,sigma2,idx_obs,idx_miss,
                                        eye_m=None,eye_L=None):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    p,L = W.shape
    W_obs = W[idx_obs==1]
    W_miss = W[idx_miss==1]

    if eye_m is None:
        eye_m=tf.constant(np.eye(int(np.sum(idx_miss))).astype(np.float32))
    if eye_L is None:
        eye_L = tf.constant(np.eye(L).astype(np.float32))
    
    sigIWoWo = sigma2*eye_L + tf.matmul(tf.transpose(W_obs),W_obs)
    ending = tf.linalg.solve(sigIWoWo,tf.transpose(W_obs))
    coefs = tf.matmul(tf.transpose(ending),tf.transpose(W_miss))
    inner = eye_L - tf.matmul(ending,W_obs)

    prec = tf.matmul(W_miss,tf.matmul(inner,tf.transpose(W_miss)))
    Sig = sigma2*eye_m + prec
    return coefs,Sig

def mvn_conditional_params_diagLinear_tf(X,W,sigma2,idx_obs,idx_miss=None):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    pass

def mvn_conditional_loglikelihood_isoLinear_tf(X_obs,X_miss,W,sigma2,
                        idx_obs,idx_miss,eye_L=None,eye_m=None):
    '''
    This computes the marginal likelihood of X_miss|X_obs  under model

    p(X) = N(0,WW^T + \sigma^2I)

    Uses woodbury matrix identity to make very quick

    Parameters
    ----------

    Returns
    -------
    log_likelihood : tf.Float
        The average log likelihood 
    '''
    p,L = W.shape
    q,r = int(np.sum(idx_miss)),int(np.sum(idx_obs))
    W_obs = W[idx_obs==1]
    W_miss = W[idx_miss==1]

    if eye_m is None:
        eye_m=tf.constant(np.eye(q).astype(np.float32))
    if eye_L is None:
        eye_L = tf.constant(np.eye(L).astype(np.float32))
    
    sigIWoWo = sigma2*eye_L + tf.matmul(tf.transpose(W_obs),W_obs)
    ending = tf.linalg.solve(sigIWoWo,tf.transpose(W_obs))

    #Extract the mean
    coefs = tf.matmul(tf.transpose(ending),tf.transpose(W_miss))
    X_hat = tf.matmul(X_obs,coefs)

    C = eye_L - tf.matmul(ending,W_obs)
    C_inv = tf.linalg.inv(C)
    WmTWm = tf.matmul(tf.transpose(W_miss),W_miss)
    CWTW = sigma2*C_inv + WmTWm
    SigInv = (eye_m - tf.matmul(W_miss,tf.linalg.solve(CWTW,tf.transpose(W_miss))))/sigma2

    diff = X_miss - X_hat

    logDetSigma = (tf.linalg.logdet(CWTW/sigma2) + tf.linalg.logdet(C) + 
                                    q*tf.math.log(sigma2))

    term1 = q*oh*pi2
    term2 = oh*logDetSigma

    quad_init = tf.matmul(diff,SigInv)
    diff_big = tf.multiply(diff,quad_init)
    row_sum = tf.reduce_sum(diff_big,axis=1)
    quad = tf.reduce_mean(row_sum)
    term3 = oh*quad

    log_likelihood = term1 + term2 + term3
    return log_likelihood

def mvn_conditional_loglikelihood_isoLinear_split_tf(X_obs,X_miss,W_obs,
                            W_miss,sigma2,eye_L=None,eye_m=None):
    '''
    This computes the marginal likelihood of X_miss|X_obs  under model

    p(X) = N(0,WW^T + \sigma^2I)

    Uses woodbury matrix identity to make very quick

    Parameters
    ----------

    Returns
    -------
    log_likelihood : tf.Float
        The average log likelihood 
    '''
    r,L = W_obs.shape
    q,_ = W_miss.shape

    if eye_m is None:
        eye_m=tf.constant(np.eye(q).astype(np.float32))
    if eye_L is None:
        eye_L = tf.constant(np.eye(L).astype(np.float32))
    
    sigIWoWo = sigma2*eye_L + tf.matmul(tf.transpose(W_obs),W_obs)
    ending = tf.linalg.solve(sigIWoWo,tf.transpose(W_obs))

    #Extract the mean
    coefs = tf.matmul(tf.transpose(ending),tf.transpose(W_miss))
    X_hat = tf.matmul(X_obs,coefs)

    C = eye_L - tf.matmul(ending,W_obs)
    C_inv = tf.linalg.inv(C)
    WmTWm = tf.matmul(tf.transpose(W_miss),W_miss)
    CWTW = sigma2*C_inv + WmTWm
    SigInv = (eye_m - tf.matmul(W_miss,tf.linalg.solve(CWTW,tf.transpose(W_miss))))/sigma2

    diff = X_miss - X_hat

    logDetSigma = (tf.linalg.logdet(CWTW/sigma2) + tf.linalg.logdet(C) + 
                                    q*tf.math.log(sigma2))

    term1 = q*oh*pi2
    term2 = oh*logDetSigma

    quad_init = tf.matmul(diff,SigInv)
    diff_big = tf.multiply(diff,quad_init)
    row_sum = tf.reduce_sum(diff_big,axis=1)
    quad = tf.reduce_mean(row_sum)
    term3 = oh*quad

    log_likelihood = term1 + term2 + term3
    return log_likelihood


def mvn_conditional_loglikelihood_diagLinear_tf(X,W,sigma2,idx_obs):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    pass

