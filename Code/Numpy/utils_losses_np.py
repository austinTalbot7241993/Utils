'''

Methods:
generalized_kl_np(X,Y)
    This implements the I-divergence used in a non-negative factorization.

alpha_divergence_np(X,Y,alpha)
    Implements the alpha-divergence used in a non-negative factorization.

beta_divergence_np(X,Y,beta)
    This implements the beta-divergence used in a non-negative factorization

itakuraSaito_np(X,Y)
    This implements the IS-divergence used in a non-negative factorization.

Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import numpy.linalg as la
from utils_activations_np import sigmoid_safe_np

EPSILON = np.finfo(np.float32).eps

def _convertXY(X,Y):
    indices = X > EPSILON
    X_data = X[indices]
    Y_data = Y[indices]
    Y_data[Y_data==0] = EPSILON
    return X_data,Y_data

################################################
################################################
##                                            ##
##  Non-negative matrix factorization losses  ##
##                                            ##
################################################
################################################

def generalized_kl_np(X,Y):
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
    loss : float
        The average loss 
    '''
    X_data,Y_data = _convertXY(X,Y)
    X_sum = X_data.sum()
    Y_sum = Y_data.sum()
    logXY = np.log(X_data/Y_data)
    XlogXY = X_data*logXY
    loss = XlogXY.sum() - X_sum + Y_sum
    return loss

def alpha_divergence_np(X,Y,alpha):
    '''
    Implements the alpha-divergence used in a non-negative factorization.

    d_a = 1/(a(a-1))x^a y^(1-a) - a*x + (a-1)*y

    Parameters 
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    alpha : float
        The exponent 

    Returns
    -------
    loss : float
        The average loss 
    '''
    X_data,Y_data = _convertXY(X,Y)
    VaY1_a = X_data**alpha * Y**(1-alpha)
    aV = alpha*X_data
    a_1Y = (alpha-1)*Y_data
    interior_sum = VaY1_a.sum() - aV.sum() + a_1Y.sum()
    loss = 1/(alpha*(alpha-1))*interior_sum
    return loss

def beta_divergence_np(X,Y,beta):
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
    loss : float
        The average loss
    '''
    X_data,Y_data = _convertXY(X,Y)
    sum_Y_beta = np.sum(Y_data**beta)
    sum_X_Y = np.dot(X_data,Y_data**(beta-1))
    loss = (X_data**beta).sum() - beta*sum_X_Y
    loss += sum_Y_beta*(beta-1)
    loss /= beta*(beta-1)
    return loss

def itakuraSaito_np(X,Y):
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
    loss : float
        The average loss
    '''
    X_data,Y_data = _convertXY(X,Y)
    div = X_data / Y_data
    term1 = np.sum(div)
    term2 = np.product(X.shape)
    term3 = np.sum(np.log(div))
    loss = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))
    return loss

def phiDivergence_np(X,Y,phi):
    '''
    This implements the phi-divergence used in a non-negative factorization.


    Parameters
    ----------
    X : array-like,(N,p)
        The true values

    Y : array-like,(N,p)
        The predicted values

    Returns
    -------
    loss : float
        The average loss
    '''
    pass


###############################
###############################
##                           ##
##  Loss truncation methods  ##
##                           ##
###############################
###############################

def bump_truncation_np(loss,k=1.0,c=1.0):
    '''
    This uses an exponential to truncate, relatively wide

    loss_trunc = exp(loss/c)*k

    Parameters
    ----------
    loss : np.float
        The original loss function

    k : np.float
        The maximum value of truncated loss

    c : np.float   
        How quickly loss saturates

    Returns
    -------
    truncated_loss : np.float
        The loss truncated between 0 and k
    '''
    exponent = -1/np.abs(loss/c)
    truncated_loss = np.exp(exponent)*k
    truncated_loss[loss==0] = 0
    return truncated_loss

def sigmoid_truncation_np(x,k=1.0,c=1.0):
    '''
    This uses a logit to truncate, relatively sharp

    loss_trunc = (2*exp(c*loss)/(1+exp(c*loss)-1)*k

    Parameters
    ----------
    loss : np.float
        The original loss function

    k : np.float
        The maximum value of truncated loss

    c : np.float   
        How quickly loss saturates

    Returns
    -------
    truncated_loss : np.float
        The loss truncated between 0 and k
    '''
    sig = sigmoid_safe_np(loss)
    truncated_loss = k*(2*sig-1)
    return truncated_loss

def flat_truncation_np(loss,k=1.0):
    '''
    This uses an exponential to truncate, relatively wide

    loss_trunc = minimum(loss,k)

    Parameters
    ----------
    loss : np.float
        The original loss function

    k : np.float
        The maximum value of truncated loss

    Returns
    -------
    truncated_loss : np.float
        The loss truncated between 0 and k
    '''
    truncated_loss = np.minimum(loss,k)
    return truncated_loss

