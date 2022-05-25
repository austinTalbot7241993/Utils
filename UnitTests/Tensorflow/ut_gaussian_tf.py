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
from utils_gaussian_tf import *

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

np.set_printoptions(suppress=True)

rand.seed(1993)

def test_mvn_loglikelihood_tf():
    print_mtm('mvn_loglikelihood_tf')

    W = rand.randn(5,2)
    cov_test = 5*np.eye(5) + np.dot(W,W.T)
    mvn2 = st.multivariate_normal(np.zeros(5),cov_test)
    
    xx = rand.randn(2,5).astype(np.float32)
    mn = np.mean(mvn2.logpdf(xx))
    cov_test_const = tf.constant(cov_test.astype(np.float32))
    tf_val = mvn_loglikelihood_tf(xx,cov_test_const)
    tolerance(np.abs(mn-tf_val.numpy()),1e-5,'Random 5d check')
    
def test_mvn_sample_tf():
    print_mtm('mvn_sample_tf')
    N = 4000000
    p = 5
    Sigma = invwishart(p+2,np.eye(p)).rvs()
    Sigma = Sigma.astype(np.float32)
    mean = rand.randn(p).astype(np.float32)

    mean_tf = np.zeros((N,p))
    for i in range(N):
        mean_tf[i] = mean
    mean_tf = mean_tf.astype(np.float32)

    X_tf = mvn_sample_tf(mean_tf,Sigma)
    X_tf_np = X_tf.numpy()

    mean_tf = np.mean(X_tf_np,axis=0)
    mdiff = np.mean(np.abs(mean-mean_tf))

    tolerance(mdiff,1e-3,'Mean 1d check')

    cov_tf = np.cov(X_tf_np.T)

    cdiff = np.mean(np.abs(Sigma-cov_tf))
    tolerance(cdiff,1e-3,'Covariance 1d check')

    N2 = 40000
    Sigma_multi = np.zeros((N2,p,p))
    mean_multi = np.zeros((N2,p))
    for i in range(N2):
        Sigma_multi[i] = Sigma
        mean_multi[i] = mean
    mean_multi = mean_multi.astype(np.float32)
    Sigma_multi = Sigma_multi.astype(np.float32)

    X_tf = mvn_sample_tf(mean_multi,Sigma_multi)
    X_tf_np = X_tf.numpy()

    mean_tf = np.mean(X_tf_np,axis=0)
    mdiff = np.mean(np.abs(mean-mean_tf))
    tolerance(mdiff,1e-1,'Mean check')

    cov_tf = np.cov(X_tf_np.T)
    cdiff = np.mean(np.abs(Sigma-cov_tf))
    tolerance(cdiff,1e-2,'Covariance check')

def test_mvn_loglikelihood_mean_tf():
    print_mtm('mvn_loglikelihood_mean_tf')
    W = rand.randn(5,2)
    cov_test = 5*np.eye(5) + np.dot(W,W.T)
    mean = rand.randn(5).astype(np.float32)
    mvn2 = st.multivariate_normal(mean,cov_test)
    
    xx = rand.randn(2,5).astype(np.float32)
    mn = np.mean(mvn2.logpdf(xx))
    cov_test_const = tf.constant(cov_test.astype(np.float32))
    tf_val = mvn_loglikelihood_mean_tf(xx,mean,cov_test_const)
    tolerance(np.abs(mn-tf_val.numpy()),1e-5,'Random 5d check')

def test_mvn_conditional_tf():
    print_mtm('mvn_conditional_tf')
    myDict = pickle.load(open('../Data/MVN_Conditional_data.p','rb'))
    idxs = myDict['idxs']
    Sigma = myDict['Sigma'].astype(np.float32)
    x_test = myDict['x_test'].astype(np.float32)
    mu_0 = myDict['condMu0'].astype(np.float32)
    mu_1 = myDict['condMu1'].astype(np.float32)
    CV_0 = myDict['condVar0'].astype(np.float32)
    CV_1 = myDict['condVar1'].astype(np.float32)
    mu_bar,Sigma_bar = mvn_conditional_tf(Sigma,idxs)
    mb_np = mu_bar.numpy()
    Sb_np = Sigma_bar.numpy()

    mu_pred = np.dot(mb_np,x_test)
    mu_diff = np.abs(mu_0-mu_pred)
    tolerance(np.sum(mu_diff),1e-5,'Conditional mean check')
    c_diff = np.abs(Sb_np-CV_0)
    tolerance(np.sum(c_diff),1e-4,'Conditional variance check')

def test_mvn_conditional_PPCA_tf():
    print_mtm('mvn_conditional_PPCA_tf')
    p = 10
    L = 3
    W = rand.randn(p,L)
    Sigma = np.dot(W,W.T) + 10*np.eye(p)
    W = W.astype(np.float32)
    Sigma = Sigma.astype(np.float32)

    idxs = np.zeros(p)
    idxs[:6] = 1


def test_mvn_conditional_mean_tf():
    print_mtm('mvn_conditional_mean_tf')

def test_mvn_kl_batch_standard_tf():
    print_mtm('mvn_kl_batch_standard_tf')
    N = 20
    p = 5
    mu = rand.randn(N,p).astype(np.float32)

    logvar = np.zeros((N,p)).astype(np.float32)
    Sigma = np.eye(p).astype(np.float32)
    baseline_divergence = mvn_kl_batch_diag_tf(mu,logvar)
    test_divergence = mvn_kl_batch_standard_tf(mu,Sigma)
    bd = baseline_divergence.numpy()
    td = test_divergence.numpy()
    diff = np.abs(bd-td)
    tolerance(diff,1e-6,'Identity KL divergence check')

    logvar = np.ones((N,p)).astype(np.float32)
    Sigma = np.exp(1)*np.eye(p).astype(np.float32)
    baseline_divergence = mvn_kl_batch_diag_tf(mu,logvar)
    test_divergence = mvn_kl_batch_standard_tf(mu,Sigma)
    bd = baseline_divergence.numpy()
    td = test_divergence.numpy()
    diff = np.abs(bd-td)
    tolerance(diff,1e-6,'Constant mul KL divergence check')

def test_mvn_loglikelihood_iso_tf():
    print_mtm('mvn_loglikelihood_iso_tf')
    N = 10
    p = 8
    X = rand.randn(10,8).astype(np.float32)
    sigma2 = 5.0
    Sigma = sigma2*np.eye(p).astype(np.float32)

    llike_full = mvn_loglikelihood_tf(X,Sigma)
    llike_iso = mvn_loglikelihood_iso_tf(X,sigma2)
    diff = np.abs(llike_full.numpy()-llike_iso.numpy())
    tolerance(diff,1e-6,'mvn_loglikelihood_iso check')

if __name__ == "__main__":
    test_mvn_loglikelihood_tf()
    #test_mvn_sample_tf()
    test_mvn_loglikelihood_mean_tf()
    test_mvn_conditional_tf()
    test_mvn_conditional_PPCA_tf()
    test_mvn_conditional_mean_tf()
    test_mvn_kl_batch_standard_tf()
    test_mvn_loglikelihood_iso_tf()
