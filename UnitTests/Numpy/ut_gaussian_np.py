import numpy as np
import numpy.random as rand
import numpy.linalg as la
from scipy import stats as st
#import tensorflow as tf
import pickle

import sys,os
sys.path.append('../../Code/Miscellaneous')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
sys.path.append('/home/austin/Utilities/Code/Numpy')
sys.path.append('/Users/austin/Utilities/Code/Tensorflow')
sys.path.append('/home/austin/Utilities/Code/Tensorflow')
from utils_gaussian_np import mvn_loglikelihood_np
from utils_gaussian_np import mvn_conditional_distribution_np
from scipy.stats import invwishart
from utils_unitTest import tolerance,greater_than
#from utils_gaussian_tf import mvn_conditional_tf


rand.seed(1993)


def test_mvn_loglikelihood_np():
    print('################################')
    print('# Testing mvn_loglikelihood_np #')
    print('################################')
    mean = rand.randn(2)
    cov = np.eye(2)
    mvn = st.multivariate_normal(mean,cov)
    spy = mvn.logpdf(np.zeros(2))
    mn = mvn_loglikelihood_np(np.zeros((1,2)),cov)

    p = 8
    iw = invwishart(df=p+2, scale=np.eye(p))
    cov = iw.rvs()
    mvn = st.multivariate_normal(np.zeros(p),cov)
    X = rand.randn(5,p)
    mn = np.mean(mvn.logpdf(X))
    spy = mvn_loglikelihood_np(X,cov)
    tolerance(np.abs(mn-spy),1e-4,'Random 10d check')

def test_mvn_conditional_distribution_np():
    print('###########################################')
    print('# Testing mvn_conditional_distribution_np #')
    print('###########################################')
    myDict = pickle.load(open('../Data/MVN_Conditional_data.p','rb'))
    idxs = myDict['idxs']
    Sigma = myDict['Sigma'].astype(np.float32)
    x_test = myDict['x_test'].astype(np.float32)
    mu_0 = myDict['condMu0'].astype(np.float32)
    mu_1 = myDict['condMu1'].astype(np.float32)
    CV_0 = myDict['condVar0'].astype(np.float32)
    CV_1 = myDict['condVar1'].astype(np.float32)

    mu_bar,Sigma_bar = mvn_conditional_distribution_np(Sigma,idxs)

    mu_pred = np.dot(mu_bar,x_test)
    mu_diff = np.abs(mu_pred-mu_0)
    tolerance(np.sum(mu_diff),1e-5,'Conditional mean check')
    c_diff = np.abs(CV_0-Sigma_bar)
    tolerance(np.sum(c_diff),1e-4,'Conditional variance check')
    
'''
def test_eval_nll_likelihood_cov_nomean():
    print('##########################################')
    print('# Testing eval_nll_likelihood_cov_nomean #')
    print('##########################################')
    mean = np.zeros(2)
    cov = np.eye(2).astype(np.float32)
    mvn = st.multivariate_normal(mean,cov)
    spy = mvn.logpdf(np.zeros(2))
    mn = eval_nll_likelihood_cov_nomean(np.zeros((1,2)),cov)
    message(np.abs(mn-spy),1e-7,'Origin 2d check')

    xx = rand.randn(4,2).astype(np.float32)
    mn = np.mean(mvn.logpdf(xx))
    spy = eval_nll_likelihood_cov_nomean(xx,cov)
    message(np.abs(mn-spy),1e-7,'Random 2d check')

    W = rand.randn(5,2)
    cov_test = 5*np.eye(5) + np.dot(W,W.T)
    mvn2 = st.multivariate_normal(np.zeros(5),cov_test)
    cov_test = cov_test
    mn = mvn2.logpdf(np.zeros(5))
    spy = eval_nll_likelihood_cov_nomean(np.zeros((1,5)),cov_test)
    message(np.abs(mn-spy),1e-7,'Origin 5d check')

    xx = rand.randn(2,5).astype(np.float32)
    mn = np.mean(mvn2.logpdf(xx))
    spy = eval_nll_likelihood_cov_nomean(xx,cov_test)
    message(np.abs(mn-spy),1e-7,'Random 5d check')
'''
if __name__ == "__main__":
    test_mvn_loglikelihood_np()
    test_mvn_conditional_distribution_np()
