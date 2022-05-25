import numpy as np
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st
import tensorflow as tf
import tensorflow_probability as tfp

import sys,os

sys.path.append('/Users/austin/Utilities/Code/Numpy')
from utils_activations_np import softplus_inverse_np
from utils_activations_np import softplus_np

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

rand.seed(1993)

def mat_err(A1,A2):
    difference = np.abs(A1-A2)
    return np.sum(difference)

def test_softplus_np():
    print_mtm('softplus_np')
    
    X = rand.randn(10,5)
    X[:,0] = 10000000*X[:,0]
    Y_tf = tf.nn.softplus(X)
    y_tf = Y_tf.numpy()
    y_np = softplus_np(X)
    err = mat_err(y_tf,y_np)
    message(err,1e-8,'softplus_np check')

def test_softplus_inverse_np():
    print_mtm('softplus_inverse_np')
    X = np.abs(rand.randn(10,5))
    X[:,0] = X[:,0]*1e-20
    Y_tf = tfp.math.softplus_inverse(X)
    y_tf = Y_tf.numpy()
    y_np = softplus_inverse_np(X)
    err = mat_err(y_tf,y_np)
    message(err,1e-8,'softplus_inverse_np check')

if __name__ == "__main__":
    test_softplus_inverse_np()
    test_softplus_np()
