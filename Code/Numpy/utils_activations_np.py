'''

'''
import numpy as np

def softplus_inverse_np(x):
    '''
    Computes the inverse of the softplus activation of x in a 
    numerically stable way
    y = np.log(np.exp(x) - 1)

    Parameters 
    ----------
    x : np.array
        Original array
    Returns
    -------
    x : np.array
        Transformed array
    '''
    threshold = np.log(np.finfo(x.dtype).eps) + 2. 
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = np.log(x)
    too_large_value = x
    y = x + np.log(-(np.exp(-np.abs(x))-1))
    y[is_too_small] = too_small_value[is_too_small]
    y[is_too_large] = too_large_value[is_too_large]
    return y

def softplus_np(x):
    '''
    Computes the softplus activation of x in a numerically stable way
    y = np.log(np.exp(y) + 1)

    Parameters 
    ----------
    x : np.array
        Original array
    Returns
    -------
    x : np.array
        Transformed array
    '''
    y = np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0)
    return y

def sigmoid_safe_np(x):
    '''


    '''
    return np.exp(-np.logaddexp(0,-x))
