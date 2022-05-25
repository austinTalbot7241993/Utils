'''

'''
import numpy as np
import numpy.random as rand
import numpy.linalg as la
from scipy import stats as st
import sys,os
from numpy.linalg import slogdet

sys.path.append('..')
sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

sys.path.append('../../Code/Numpy')
from utils_matrix_np import *

rand.seed(1993)

def mat_err(A1,A2):
    difference = np.abs(A1-A2)
    return np.sum(difference)

def test_woodbury_inverse_np(time=True):
    print_mtm('woodbury_inverse_np')

    p = 10
    q = 5
    A = np.diag(rand.randn(p))
    U = rand.randn(p,q)
    C = np.diag(rand.randn(q))
    V = rand.randn(q,p)
    UCV = np.dot(U,np.dot(C,V))
    
    AUCV = la.inv(A+UCV)
    woodbury = woodbury_inverse_np(A,U,C,V)
    diff = mat_err(AUCV,woodbury)
    message(diff,1e-5,'woodbury_inverse_np check')

    if time:
        p = 400
        q = 10 
        A = np.eye(p)
        U = rand.randn(p,q)
        C = np.eye(q)
        V = rand.randn(q,p)
        T = A + U.dot(np.dot(C,V))
        def func_numpy():
            AUCV = la.inv(T)
        def func_woodbury():
            AUCV = woodbury_inverse_np(A,U,C,V)

        time_numpy = time_method(func_numpy,num=30)
        time_woodbury = time_method(func_woodbury,num=30)
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

    print('')

def test_woodbury_inverse_sym_np(time=True):
    print_mtm('woodbury_inverse_sym_np')
    p = 10
    q = 5
    A = np.diag(rand.randn(p))
    W = rand.randn(p,q)
    C = A + np.dot(W,W.T)
    Cinv = la.inv(C)
    Ainv = la.inv(A)
    woodbury = woodbury_inverse_sym_np(Ainv,W)
    diff = mat_err(Cinv,woodbury)
    message(diff,1e-5,'woodbury_inverse_sym_np check')

    if time:
        p = 1000
        q = 10
        A = np.diag(10*rand.randn(p))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)
        eye2 = np.eye(q)
        def func_numpy():
            Cinv = la.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_np(Ainv,W)
        def func_woodbury2():
            Winv = woodbury_inverse_sym_np(Ainv,W,eye2)
        time_numpy = time_method(func_numpy,num=50)
        time_woodbury = time_method(func_woodbury,num=50)
        time_woodbury2 = time_method(func_woodbury2,num=50)
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)
        print('Woodbury time %0.3f'%time_woodbury2)

    print('')

def test_subset_square_matrix_np():
    print_mtm('subset_square_matrix_np')
    p = 10
    A = rand.randn(p,p)
    idxs = np.zeros(p)
    idxs[:6] = 1
    idxs2 = np.zeros(p)
    idxs2[3:] = 1
    A_sub1 = subset_square_matrix_np(A,idxs)
    A_sub2 = subset_square_matrix_np(A,idxs2)

    #Compute manually
    manual1_11 = A[idxs==1]
    Am_sub1 = manual1_11[:,idxs==1]

    manual2_11 = A[idxs2==1]
    Am_sub2 = manual2_11[:,idxs2==1]

    e1 = mat_err(A_sub1,Am_sub1)
    e2 = mat_err(A_sub2,Am_sub2)
    tolerance(e1,1e-8,'Matrix subset 1 check')
    tolerance(e2,1e-8,'Matrix subset 2 check')
    print('')

def test_woodbury_sldet_np(time=True):
    print_mtm('woodbury_sldet_np')
    p = 100
    q = 10 
    A = np.eye(p)
    U = rand.randn(p,q)
    C = np.eye(q)
    V = rand.randn(q,p)
    T = A + U.dot(np.dot(C,V))
    sn,ldet = slogdet(T)

    sn_test,ldet_test = woodbury_sldet_np(A,U,C,V)
    tolerance(sn-sn_test,1e-8,'woodbury_sldet_np sign check')
    tolerance(ldet_test-ldet,1e-4,'woodbury_sldet_np ldet check')

    if time:
        p = 1000
        q = 10 
        A = np.eye(p)
        U = rand.randn(p,q)
        C = np.eye(q)
        V = rand.randn(q,p)
        T = A + U.dot(np.dot(C,V))
        def func_numpy():
            sn,ldet = slogdet(T)
        def func_woodbury():
            sn_test,ldet_test = woodbury_sldet_np(A,U,C,V)

        time_numpy = time_method(func_numpy,num=100)
        time_woodbury = time_method(func_woodbury,num=100)
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

        p = 2000
        q = 3 
        A = np.eye(p)
        U = rand.randn(p,q)
        C = np.eye(q)
        V = rand.randn(q,p)
        T = A + U.dot(np.dot(C,V))
        def func_numpy():
            sn,ldet = slogdet(T)
        def func_woodbury():
            woodbury_sldet_np(A,U,C,V)
            #sn_test,ldet_test = woodbury_sldet_np(A,U,C,V)
        time_numpy = time_method(func_numpy,num=30)
        time_woodbury = time_method(func_woodbury,num=30)
        print('---------')
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

    print('')

def test_woodbury_sldet_sym_np(time=True):
    print_mtm('woodbury_sldet_sym_np')
    p = 10
    q = 5
    A = np.diag(np.abs(rand.randn(p)))
    W = rand.randn(p,q)
    C = A + np.dot(W,W.T)
    wldet = woodbury_sldet_sym_np(A,W)
    sn,nldet = slogdet(C)
    diff = np.abs(wldet-nldet)
    message(diff,1e-5,'woodbury_sldet_sym_np check')

    if time:
        p = 1000
        q = 10
        A = np.diag(10*rand.randn(p))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)
        eye2 = np.eye(q)
        def func_numpy():
            Cinv = la.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_np(A,W)
        time_numpy = time_method(func_numpy,num=50)
        time_woodbury = time_method(func_woodbury,num=50)
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

    print('')

def test_quadratic_form_np():
    print_mtm('quadratic_form_np')
    p = 5
    mu = rand.randn(p)
    w = rand.rand(p,1)
    cov = np.dot(w,w.T) + np.eye(p)
    w2 = rand.rand(p,1)
    Lamb = np.dot(w2,w2.T) + np.eye(p)
    X = rand.multivariate_normal(mu,cov,size=10000000)

    mean_sim = evaluate_quad_np(X,Lamb)
    mean_calc = quadratic_form_np(Lamb,mu,cov)

    diff = np.abs(mean_calc-mean_sim)
    tolerance(diff,1e-4,'quadratic_form_np check')
    print('')

def test_quadratic_form_batch_np():
    print_mtm('quadratic_form_batch_np')
    p = 5
    N = 10000000
    Mu = rand.randn(N,p)
    w = rand.rand(p,1)
    cov = np.dot(w,w.T) + np.eye(p)
    w2 = rand.rand(p,1)
    Lamb = np.dot(w2,w2.T) + np.eye(p)

    X = rand.multivariate_normal(np.zeros(p),cov,size=N) + Mu

    mean_sim = evaluate_quad_np(X,Lamb)
    mean_calc = quadratic_form_batch_np(Lamb,Mu,cov)

    diff = np.abs(mean_calc-mean_sim)
    tolerance(diff,1e-3,'quadratic_form_batch_np check')
    print('')
    

if __name__ == "__main__":
    print_ftm('utils_matrix_np')
    time = False
    test_woodbury_inverse_np(time=time)
    test_woodbury_inverse_sym_np(time=time)
    test_subset_square_matrix_np()
    test_woodbury_sldet_np(time=time)
    test_quadratic_form_np()
    test_quadratic_form_batch_np()
    print('')












