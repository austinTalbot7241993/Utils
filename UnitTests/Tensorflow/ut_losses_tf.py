import numpy as np
import numpy.random as rand
import numpy.linalg as la
from scipy import stats as st
import tensorflow as tf
import pickle

import sys,os
sys.path.append('../../Code/Tensorflow')
sys.path.append('../../Code/Numpy')
from utils_losses_tf import loss_generalized_kl_tf
from utils_losses_tf import loss_alpha_divergence_tf
from utils_losses_tf import loss_beta_divergence_tf
from utils_losses_tf import loss_itakuraSaito_tf

from utils_losses_np import generalized_kl_np
from utils_losses_np import alpha_divergence_np
from utils_losses_np import beta_divergence_np
from utils_losses_np import itakuraSaito_np

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

rand.seed(1993)

from sklearn.decomposition._nmf import _beta_divergence

def generateData(N,p,L,zeros=True):
    X = np.abs(rand.randn(N,p))
    S = np.abs(rand.randn(N,L))
    W = np.abs(rand.randn(L,p))
    if zeros:
        S[0] = 0
        X[0,0] = 0
    Y = np.dot(S,W)
    return X,Y

def test_generalized_kl_tf():
    print_mtm('generalized_kl_tf')
    X,Y = generateData(10,30,5)
    loss_mine = generalized_kl_np(X,Y)
    loss_tf = loss_generalized_kl_tf(X,Y)
    loss_diff = np.abs(loss_mine-loss_tf.numpy())
    tolerance(loss_diff,1e-4,'Generalized kl check')

def test_alpha_divergence_tf():
    print_mtm('alpha_divergence_tf')
    X,Y = generateData(10,30,5)
    loss_mine = loss_alpha_divergence_tf(X,Y)
    loss_tf = generalized_kl_tf(X,Y)
    loss_diff = np.abs(loss_mine-loss_tf.numpy())
    tolerance(loss_diff,1e-4,'Generalized kl check')
    

def test_beta_divergence_tf():
    print_mtm('beta_divergence_tf')

def test_itakuraSaito_tf():
    print_mtm('itakuraSaito_tf')
    X,Y = generateData(10,30,5,zeros=False)
    loss_mine = itakuraSaito_np(X,Y)
    loss_tf = loss_itakuraSaito_tf(X,Y)
    loss_diff = np.abs(loss_mine-loss_tf.numpy())
    tolerance(loss_diff,1e-3,'IS check')

def test_phiDivergence_tf():
    print_mtm('phiDivergence_tf')

if __name__ == "__main__":
    print_ftm('utils_losses_tf')
    test_generalized_kl_tf()
    test_itakuraSaito_tf()











