import numpy as np
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st
from scipy.stats import invwishart
from numpy.random import multivariate_normal
import pickle

import sys,os

sys.path.append('../../Code/Tensorflow')
sys.path.append('../../Code/Miscellaneous')
from utils_ppca_tf import *
from utils_gaussian_tf import mvn_loglikelihood_tf
from utils_gaussian_tf import mvn_conditional_tf

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

np.set_printoptions(suppress=True)

rand.seed(1993)

def generateData(N=1000,L=5,p=100,sigma=1.0):
    S = rand.randn(N,L)
    W = rand.randn(L,p)
    X_noise = sigma*rand.randn(N,p)
    X_hat = np.dot(S,W)
    X = X_hat + X_noise

    cov = sigma**2*np.eye(p) + np.dot(W.T,W)

    X = X.astype(np.float32)
    cov = cov.astype(np.float32)
    W = W.T.astype(np.float32)

    return X,W,X_hat,X_noise,cov
    

def test_mvn_loglikelihood_isoLinear_tf(time=True):
    print_mtm('mvn_loglikelihood_isoLinear_tf')
    sigma = 2.0
    L,p = 5,100
    X,W,X_hat,X_noise,cov = generateData(L=L,p=p,sigma=sigma)
    eye = tf.constant(np.eye(L).astype(np.float32))
    eye2 = tf.constant(np.eye(p).astype(np.float32))
    X2 = tf.constant(X)
    W2 = tf.constant(W)
    cov2 = tf.constant(cov)

    likelihood_true = mvn_loglikelihood_tf(X,cov2)
    likelihood_iso = mvn_loglikelihood_isoLinear_tf(X,W2,sigma**2,
                                            eye=eye,eye2=eye2)
    diff = likelihood_true  - likelihood_iso 
    tolerance(diff,1e-3,'mvn_loglikelihood_isoLinear_tf check')

    def time_block(N=1000,L=5,p=100):
        X,W,X_hat,X_noise,cov = generateData(L=L,p=p,sigma=sigma)
        eye = tf.constant(np.eye(L).astype(np.float32))
        eye2 = tf.constant(np.eye(p).astype(np.float32))
        X2 = tf.constant(X)
        W2 = tf.constant(W)
        cov2 = tf.constant(cov)
        def func_nonopt():
            like = mvn_loglikelihood_tf(X,cov)
        def func_partopt():
            like = mvn_loglikelihood_tf(X2,cov2)
        def func_optpart():
            like = mvn_loglikelihood_isoLinear_tf(X,W2,sigma**2)
        def func_optfull():
            like = mvn_loglikelihood_isoLinear_tf(X2,W2,sigma**2,
                                                    eye=eye,eye2=eye2)
        time_nonopt = time_method(func_nonopt,num=100)
        time_partopt = time_method(func_partopt,num=100)
        time_optpart = time_method(func_optpart,num=100)
        time_optfull = time_method(func_optfull,num=100)
        print('Original time %0.3f'%time_nonopt)
        print('Tf constant time %0.3f'%time_partopt)
        print('Woodbury time %0.3f'%time_optpart)
        print('Full opt time %0.3f'%time_optfull)
            
    if time:
        print('N=1000,L=5,p=100')
        time_block()
        print('N=1000,L=5,p=100')
        time_block()
        print('N=100000,L=5,p=440')
        time_block(N=100000,L=5,p=440)
        print('N=100000,L=5,p=1000')
        time_block(N=100000,L=5,p=1000)
    print('')
    print('')
    print('')
    print('')

def test_mvn_loglikelihood_diagLinear_tf(time=True):
    print_mtm('mvn_loglikelihood_diagLinear_tf')

    print('')
    print('')
    print('')
    print('')


def test_mvn_conditional_params_isoLinear_tf(time=True):
    print_mtm('mvn_conditional_params_isoLinear_tf')
    sigma = 2.0
    L,p = 5,100
    idx_obs = np.zeros(p)
    idx_obs[:45] = 1
    idx_miss = 1-idx_obs
    X,W,X_hat,X_noise,cov = generateData(L=L,p=p,sigma=sigma)
    X_obs = X[:,idx_obs==1]
    X_miss = X[:,idx_obs==1]
    
    mu_true,Sig_true = mvn_conditional_tf(cov,idx_obs)
    mu_opt,Sig_opt = mvn_conditional_params_isoLinear_tf(W,sigma**2,idx_obs,
                                                            idx_miss)
    diff = tf.reduce_mean(tf.abs(tf.transpose(mu_true)-mu_opt))
    tolerance(diff,1e-3,'mvn_conditional_params_isoLinear_tf mean check')
    diff = tf.reduce_mean(tf.abs(Sig_opt-Sig_true))
    tolerance(diff,1e-3,'mvn_conditional_params_isoLinear_tf Cov check')

    def time_block(N=1000,L=5,p=100,n_miss=50):
        X,W,X_hat,X_noise,cov = generateData(L=L,p=p,sigma=sigma)
        idx_obs = np.zeros(p)
        idx_obs[n_miss:] = 1
        idx_miss = 1 - idx_obs
        eye_m = tf.constant(np.eye(n_miss).astype(np.float32))
        eye_L = tf.constant(np.eye(L).astype(np.float32))
        W2 = tf.constant(W)
        cov2 = tf.constant(cov)
        def func_nonopt():
            a,b = mvn_conditional_tf(cov2,idx_obs)
        def func_optpart():
            a,b = mvn_conditional_params_isoLinear_tf(W,sigma**2,idx_obs,
                                                            idx_miss)
        def func_optfull():
            a,b = mvn_conditional_params_isoLinear_tf(W,sigma**2,idx_obs,
                                          idx_miss,eye_m=eye_m,eye_L=eye_L)

        time_nonopt = time_method(func_nonopt,num=100)
        time_optpart = time_method(func_optpart,num=100)
        time_optfull = time_method(func_optfull,num=100)
        print('Original time %0.3f'%time_nonopt)
        print('no constant time %0.3f'%time_optpart)
        print('Full opt time %0.3f'%time_optfull)
            
    if time:
        print('N=1000,L=5,p=100')
        time_block()
        print('N=100000,L=5,p=440')
        time_block(N=100000,L=5,p=440)
        print('N=100000,L=5,p=1000')
        time_block(N=100000,L=5,p=1000)
        print('N=100000,L=20,p=440')
        time_block(N=100000,L=20,p=440)

    print('')
    print('')
    print('')
    print('')

def test_mvn_conditional_loglikelihood_isoLinear_tf(time=True):
    print_mtm('mvn_conditional_loglikelihood_isoLinear_tf')

    sigma = 2.0
    L,p = 5,100
    idx_obs = np.zeros(p)
    idx_obs[:45] = 1
    idx_miss = 1-idx_obs
    X,W,X_hat,X_noise,cov = generateData(L=L,p=p,sigma=sigma)
    X_obs = X[:,idx_obs==1]
    X_miss = X[:,idx_miss==1]

    mu_true,Sig_true = mvn_conditional_tf(cov,idx_obs)

    mean = tf.matmul(X_obs,tf.transpose(mu_true))

    diff = X_miss - mean
    cond_like = mvn_loglikelihood_tf(diff,Sig_true)

    cond_model = mvn_conditional_loglikelihood_isoLinear_tf(X_obs,X_miss,
                                    W,sigma**2,idx_obs,idx_miss)

    diff = tf.reduce_mean(tf.abs(cond_like-cond_model))
    tolerance(diff,1e-3,
                'test_mvn_conditional_loglikelihood_isoLinear_tf check')

    def time_block(N=1000,L=5,p=100,n_miss=50):
        X,W,X_hat,X_noise,cov = generateData(N=N,L=L,p=p,sigma=sigma)
        idx_obs = np.zeros(p)
        idx_obs[n_miss:] = 1
        idx_miss = 1 - idx_obs
        eye_m = tf.constant(np.eye(n_miss).astype(np.float32))
        eye_L = tf.constant(np.eye(L).astype(np.float32))
        W2 = tf.constant(W)
        cov2 = tf.constant(cov)
        X_obs = X[:,idx_obs==1]
        X_miss = X[:,idx_miss==1]
        X2_obs = tf.constant(X_obs)
        X2_miss = tf.constant(X_miss)
        def func_nonopt():
            mu_true,Sig_true = mvn_conditional_tf(cov,idx_obs)
            mean = tf.matmul(X_obs,tf.transpose(mu_true))
            diff = X_miss - mean
            cond_like = mvn_loglikelihood_tf(diff,Sig_true)
        def func_optpart():
            cond_model = mvn_conditional_loglikelihood_isoLinear_tf(X_obs,
                                    X_miss,W,sigma**2,idx_obs,idx_miss)
        def func_optfull():
            cond_model = mvn_conditional_loglikelihood_isoLinear_tf(X2_obs,
                X2_miss,W,sigma**2,idx_obs,idx_miss,eye_L=eye_L,eye_m=eye_m)

        time_nonopt = time_method(func_nonopt,num=100)
        time_optpart = time_method(func_optpart,num=100)
        time_optfull = time_method(func_optfull,num=100)
        print('Original time %0.3f'%time_nonopt)
        print('no constant time %0.3f'%time_optpart)
        print('Full opt time %0.3f'%time_optfull)

    if time:
        print('N=1000,L=5,p=100')
        time_block()
        print('N=1000,L=5,p=100')
        time_block(p=440)
        print('N=10000,L=5,p=440')
        time_block(N=10000,p=440)
        print('N=10000,L=5,p=1000')
        time_block(N=10000,p=1000)
    print('')
    print('')
    print('')
    print('')

def test_mvn_conditional_loglikelihood_diagLinear_tf(time=True):
    print_mtm('mvn_conditional_loglikelihood_diagLinear_tf')


    print('')
    print('')
    print('')
    print('')

def test_mvn_conditional_params_diagLinear_tf(time=True):
    print_mtm('mvn_conditional_params_diagLinear_tf')


    print('')
    print('')
    print('')
    print('')


if __name__ == "__main__":
    print_ftm('utils_ppca_tf')

    test_mvn_loglikelihood_isoLinear_tf(time=False)
    test_mvn_loglikelihood_diagLinear_tf()
    test_mvn_conditional_params_isoLinear_tf(time=False)
    test_mvn_conditional_loglikelihood_isoLinear_tf()
    test_mvn_conditional_loglikelihood_diagLinear_tf()
    test_mvn_conditional_params_diagLinear_tf()
