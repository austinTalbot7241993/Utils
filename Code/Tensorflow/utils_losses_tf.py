'''
This implements methods related to various loss functions, particularly 
relevant to non-negative matrix factorization

Methods:
generalized_kl_tf(X,Y)
    This implements the I-divergence used in a non-negative factorization.

alpha_divergence_tf(X,Y)
    Implements the alpha-divergence used in a non-negative factorization.

beta_divergence_tf(X,Y)
    This implements the beta-divergence used in a non-negative factorization

itakuraSaito_tf(X,Y)
    This implements the IS-divergence used in a non-negative factorization.


Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import pickle
import numpy.random as rand
from sklearn import decomposition as dp
from sklearn import linear_model as lm
from tensorflow import keras
import tensorflow as tf

EPSILON = np.finfo(np.float32).eps

def loss_cosine_similarity_tf(vec1,vec2):
    v1 = tf.squeeze(vec1)
    v2 = tf.squeeze(vec2)
    num = tf.reduce_mean(tf.multiply(v1,v2))
    denom = tf.linalg.norm(v1)*tf.linalg.norm(v2)
    return tf.abs(num)


def loss_nuclear_norm_tf(X):
    ''' 
    This computes the nuclear norm via an SVD. This is the convex relaxation
    of rank(X).

    Paramters
    ---------
    X : tf.matrix,shape=(m,n)
        The input matrix

    Returns
    -------
    norm : tf.Float
        The estimated norm
    '''
    s = tf.linalg.svd(X,compute_uv=False)
    norm = tf.reduce_sum(s)
    return norm

def loss_generalized_kl_tf(X,Y,weights=None):
    '''
    This implements the I-divergence used in a non-negative factorization.

    d(x|y) = x log(x/y) - x + y

    Parameters 
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    Returns
    -------
    loss : tf.float
        The average loss 
    '''
    X_buffered = X + EPSILON
    Y_buffered = Y + EPSILON
    X_sum = tf.reduce_sum(X_buffered)
    Y_sum = tf.reduce_sum(Y_buffered)
    logXY = tf.math.log(X_buffered/Y_buffered)
    XlogXY = tf.multiply(X_buffered,logXY)
    loss = tf.reduce_sum(XlogXY) - X_sum + Y_sum
    return loss

def loss_alpha_divergence_tf(X,Y,weights=None):
    '''
    Implements the alpha-divergence used in a non-negative factorization.

    Parameters 
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    Returns
    -------
    loss : tf.float
        The average loss 
    '''
    pass

def loss_beta_divergence_tf(X,Y,beta,weights=None):
    '''
    This implements the beta-divergence used in a non-negative factorization

    Parameters 
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    Returns
    -------
    loss : tf.float
        The average loss 
    '''
    X_buffered = X + EPSILON
    Y_buffered = Y + EPSILON
    sum_Y_beta = tf.reduce_sum(tf.math.pow(Y_buffered,beta))
    sum_X_Y = tf.reduce_sum(tf.multiply(X_buffered,
                                tf.math.pow(Y_buffered,beta-1)))
    loss_1 = tf.reduce_sum(tf.math.pow(X_buffered,beta)) - beta*sum_X_Y
    loss_2 = sum_Y_beta*(beta-1)
    loss = (loss_1 + loss_2)/(beta*(beta-1))
    return loss

def loss_itakuraSaito_tf(X,Y,weights=None):
    '''
    This implements the IS-divergence used in a non-negative factorization.

    d(x,y) = x/y - log(x/y) - 1

    Parameters 
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    Returns
    -------
    loss : tf.float
        The average loss 
    '''
    X_buffered = X + EPSILON
    Y_buffered = Y + EPSILON
    div = X_buffered/Y_buffered
    term1 = tf.reduce_sum(div)
    term2 = np.product(X.shape)
    term3 = tf.reduce_sum(tf.math.log(div))
    loss_unscaled = term1 - term2 - term3
    loss = loss_unscaled/X.shape[0]
    return loss

def L2_loss_tf(X,weights=None):
    return tf.reduce_mean(tf.square(X))

def gamma_loss_tf(x,alpha,beta,weights=None):
    return (alpha-1)*tf.math.log(x) - beta*x

def nll_gamma_tf(x,a,b,weights=None):
    '''
    The average negative log likelihood of gamma distribution
    log p(x) = alog(b) - lG(a) + (a-1)log(x) - b/x


    Parameters
    ----------
    x : array
        The points to evaluate the 

    a : float 
        Shape

    b : float
        Scale

    weights : array-like,(x.shape)
        The (optional) weights on the observations
        
    Returns
    -------
    nll : tf.Float
        Negative log likelihood
    '''
    xp = tf.squeeze(x)
    lpdf =a*tf.math.log(b) - tf.math.lgamma(a) +(a-1)*tf.math.log(xp) - b/xp
    if weights is None:
        nll = -1*tf.reduce_mean(lpdf)
    else:
        nll = -1*tf.reduce_mean(lpdf*weights)
    return nll

def nll_invgamma_tf(x,a,b,weights=None):
    '''
    The average negative log likelihood of inverse gamma distribution
    log p(x) = alog(b) - lG(a) - (a+1)log(x) - b/x

    Parameters
    ----------
    x : array
        The points to evaluate the 

    a : float 
        Shape

    b : float
        Scale

    weights : array-like,(x.shape)
        The (optional) weights on the observations
        
    Returns
    -------
    nll : tf.Float
        Negative log likelihood
    '''
    xp = tf.squeeze(x)
    lpdf =a*tf.math.log(b) - tf.math.lgamma(a) -(a+1)*tf.math.log(xp) - b/xp
    if weights is None:
        nll = -1*tf.reduce_mean(lpdf)
    else:
        nll = -1*tf.reduce_mean(lpdf*weights)
    return nll

def nll_exp_tf(x,lamb,weights=None):
    '''
    The average negative log likelihood of exponential distribution
    log p(x) = -lambda*x - log(lambda)

    Parameters
    ----------
    x : array
        The points to evaluate the 

    lamb : float
        Mean

    weights : array-like,(x.shape)
        The (optional) weights on the observations
        
    Returns
    -------
    nll : tf.Float
        Negative log likelihood
    '''
    lpdf  = -lamb*tf.squeeze(x) - tf.math.log(lamb)
    nll  -1*tf.reduce_mean(lpdf)
    return nll

def loss_Lq_tf(M,q=2,weights=None):
    '''
    Computes the Lq loss on the off-diagonal elements

    Parameters
    ----------
    M : tf.array-like(p,r)
        Matrix

    q : float>0
        The power 
    Returns
    -------
    loss : tf.Float
        The loss
    '''
    if q == 1:
        loss = tf.reduce_mean(tf.abs(M))
    elif q == 2:
        loss = tf.reduce_mean(tf.square(M))
    else:
        loss = tf.reduce_mean(tf.math.pow(tf.abs(M),q))
    return loss

def loss_offdiag_Lq_tf(Sigma,q=2):
    '''
    Computes the Lq loss on the off-diagonal elements

    Parameters
    ----------
    Sigma : tf.array-like(p,p)
        Square matrix

    q : float>0
        The power 
    Returns
    -------
    loss : tf.Float
        The loss
    '''
    D = tf.linalg.diag(tf.linalg.diag_part(Sigma))
    S_D = Sigma-D
    if q == 1:
        loss = tf.reduce_mean(tf.abs(S_D))
    elif q == 2:
        loss = tf.reduce_mean(tf.square(S_D))
    else:
        loss = tf.reduce_mean(tf.math.pow(tf.abs(S_D),q))
    return loss
    

def loss_CE_tf(Y_true,Y_est,weights=None):
    '''
    '''
    Yt = tf.squeeze(Y_true)
    Ye = tf.squeeze(Y_est)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=Yt,logits=Ye)
    loss = tf.reduce_mean(ce)
    return loss

