import numpy as np
import numpy.random as rand
import numpy.linalg as la
from scipy import stats as st
import pickle

import sys,os
sys.path.append('../../Code/Miscellaneous')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
sys.path.append('/home/austin/Utilities/Code/Numpy')
from utils_losses_np import generalized_kl_np
from utils_losses_np import alpha_divergence_np
from utils_losses_np import beta_divergence_np
from utils_losses_np import itakuraSaito_np
from utils_losses_np import phiDivergence_np

from utils_unitTest import tolerance,greater_than

rand.seed(1993)

from sklearn.decomposition._nmf import _beta_divergence

def generateData(N,p,L):
    X = np.abs(rand.randn(N,p))
    X[0,0] = 0
    S = np.abs(rand.randn(N,L))
    W = np.abs(rand.randn(L,p))
    S[0] = 0
    Y = np.dot(S,W)
    return X,Y,S,W

def test_generalized_kl_np():
    print('#############################')
    print('# Testing generalized_kl_np #')
    print('#############################')
    X,Y,S,W = generateData(10,30,5)
    loss_mine = generalized_kl_np(X,Y)
    loss_true = _beta_divergence(X,S,W,1)
    loss_diff = np.abs(loss_mine-loss_true)
    tolerance(loss_diff,1e-4,'Generalized kl check')

def test_alpha_divergence_np():
    print('###############################')
    print('# Testing alpha_divergence_np #')
    print('###############################')
    X,Y,S,W = generateData(10,30,5)
    

def test_beta_divergence_np():
    print('##############################')
    print('# Testing beta_divergence_np #')
    print('##############################')
    X,Y,S,W = generateData(10,30,5)

    beta1 = 1.6
    loss_mine = beta_divergence_np(X,Y,beta1)
    loss_true = _beta_divergence(X,S,W,beta1)
    loss_diff = np.abs(loss_mine-loss_true)
    tolerance(loss_diff,1e-4,'Beta=1.6 check')

    beta2 = 2.6
    loss_mine = beta_divergence_np(X,Y,beta2)
    loss_true = _beta_divergence(X,S,W,beta2)
    loss_diff = np.abs(loss_mine-loss_true)
    tolerance(loss_diff,1e-4,'Beta=2.6 check')

def test_itakuraSaito_np():
    print('###########################')
    print('# Testing itakuraSaito_np #')
    print('###########################')
    X,Y,S,W = generateData(10,30,5)
    loss_mine = itakuraSaito_np(X,Y)
    loss_true = _beta_divergence(X,S,W,0)
    loss_diff = np.abs(loss_mine-loss_true)
    tolerance(loss_diff,1e-4,'itakuraSaito_np check')

def test_phiDivergence_np():
    print('############################')
    print('# Testing phiDivergence_np #')
    print('############################')
    X,Y,S,W = generateData(10,30,5)


if __name__ == "__main__":
    test_generalized_kl_np()
    test_alpha_divergence_np()
    test_beta_divergence_np()
    test_itakuraSaito_np()
    test_phiDivergence_np()











