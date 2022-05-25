'''

'''
import numpy as np
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st
import sys,os
from scipy.linalg import block_diag

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import *

sys.path.append('../../Code/Tensorflow')
from utils_matrix_tf import *

rand.seed(1993)

def mat_err(A1,A2):
    difference = np.abs(A1-A2)
    return np.sum(difference)

def test_subset_matrix_tf():
    print_mtm('subset_matrix_tf and subset_square_matrix_tf')
    p = 10
    q = 6
    X = rand.randn(p,q)
    X2 = rand.randn(q,q)
    X_sub1 = X[5:,2:]
    X_sub2 = X2[2:,2:]
    idx1 = np.ones(p)
    idx1[:5] = 0
    idx2 = np.zeros(q)
    idx2[2:] = 1

    X_s_tf1 = subset_matrix_tf(X,idx1==1,idx2==1)
    X_s_tf2 = subset_square_matrix_tf(X2,idx2==1)
    diff1 = np.sum(np.abs(X_s_tf1.numpy() - X_sub1))
    diff2 = np.sum(np.abs(X_s_tf2.numpy() - X_sub2))

    tolerance(diff1,1e-6,'subset_matrix_tf check')
    tolerance(diff2,1e-6,'subset_square_matrix_tf check')
    print('')

##############################
##############################
##                          ##
## Test woodbury identities ##
##                          ##
##############################
##############################

def test_woodbury_inverse_tf(time=False):
    print_mtm('woodbury_inverse_tf')
    p = 10
    q = 5
    A = rand.randn(p,p).astype(np.float32)
    U = rand.randn(p,q).astype(np.float32)
    C = rand.randn(q,q).astype(np.float32)
    V = rand.randn(q,p).astype(np.float32)
    UCV = np.dot(U,np.dot(C,V))
    AUCV = la.inv(A+UCV)

    Atf = tf.Variable(A)
    Utf = tf.Variable(U)
    Ctf = tf.Variable(C)
    Vtf = tf.Variable(V)
    woodbury = woodbury_inverse_tf(Atf,Utf,Ctf,Vtf)
    woodbury_np = woodbury.numpy()
    diff = mat_err(AUCV,woodbury_np)
    message(diff,1e-3,'woodbury_inverse_tf check')

    if time:
        p = 3000
        q = 10
        A = tf.Variable(np.eye(p).astype(np.float32))
        U = tf.Variable(rand.randn(p,q).astype(np.float32))
        C = tf.Variable(np.eye(q).astype(np.float32))
        V = tf.Variable(rand.randn(q,p).astype(np.float32))
        T = A + tf.matmul(U,tf.matmul(C,V))
        #T = A + U.dot(np.dot(C,V))
        def func_numpy():
            AUCV = tf.linalg.inv(T)
        def func_woodbury():
            AUCV = woodbury_inverse_tf(A,U,C,V)
        time_numpy = time_method(func_numpy,num=500)
        time_woodbury = time_method(func_woodbury,num=500)
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

    print('')

def test_woodbury_inverse_sym_tf(time=False):
    print_mtm('woodbury_inverse_sym_tf')
    p = 10
    q = 5
    A = np.diag(rand.randn(p)).astype(np.float32)
    W = rand.randn(p,q).astype(np.float32)
    C = A + np.dot(W,W.T)
    Cinv = la.inv(C)
    Ainv = la.inv(A)
    Ai = tf.Variable(Ainv.astype(np.float32))
    Wi = tf.Variable(W)
    woodbury = woodbury_inverse_sym_tf(Ai,Wi)
    woodbury_np = woodbury.numpy()
    diff = mat_err(Cinv,woodbury_np)
    message(diff,1e-3,'woodbury_inverse_sym_tf check')
    prod = np.dot(C,woodbury_np)
    diff2 = mat_err(np.eye(p),prod)
    message(diff2,1e-3,'woodbury_inverse_sym_tf check2')
    prod = np.dot(C,woodbury_np)

    if time:
        p = 3000
        q = 10
        A = tf.Variable(np.diag(10*rand.randn(p)).astype(np.float32))
        Ainv = tf.Variable(np.diag(1/np.diag(A)).astype(np.float32))
        W = tf.Variable(rand.randn(p,q).astype(np.float32))
        C = A + tf.matmul(W,tf.transpose(W))
        eye2 = tf.Variable(np.eye(q).astype(np.float32))
        def func_numpy():
            Cinv = tf.linalg.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_tf(Ainv,W)
        time_numpy = time_method(func_numpy,num=500)
        time_woodbury = time_method(func_woodbury,num=500)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

        p = 30
        q = 10
        A = tf.Variable(np.diag(10*rand.randn(p)).astype(np.float32))
        Ainv = tf.Variable(np.diag(1/np.diag(A)).astype(np.float32))
        W = tf.Variable(rand.randn(p,q).astype(np.float32))
        C = A + tf.matmul(W,tf.transpose(W))
        eye2 = tf.Variable(np.eye(q).astype(np.float32))
        def func_numpy():
            Cinv = tf.linalg.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_tf(Ainv,W)
        time_numpy = time_method(func_numpy,num=500)
        time_woodbury = time_method(func_woodbury,num=500)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

        p = 40
        q = 3
        A = tf.Variable(np.diag(10*rand.randn(p)).astype(np.float32))
        Ainv = tf.Variable(np.diag(1/np.diag(A)).astype(np.float32))
        W = tf.Variable(rand.randn(p,q).astype(np.float32))
        C = A + tf.matmul(W,tf.transpose(W))
        eye2 = tf.Variable(np.eye(q).astype(np.float32))
        def func_numpy():
            Cinv = tf.linalg.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_tf(Ainv,W)
        time_numpy = time_method(func_numpy,num=500)
        time_woodbury = time_method(func_woodbury,num=500)
        #time_woodbury2 = time_method(func_woodbury2,num=50)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

        p = 8
        q = 2
        A = tf.Variable(np.diag(10*rand.randn(p)).astype(np.float32))
        Ainv = tf.Variable(np.diag(1/np.diag(A)).astype(np.float32))
        W = tf.Variable(rand.randn(p,q).astype(np.float32))
        C = A + tf.matmul(W,tf.transpose(W))
        eye2 = tf.Variable(np.eye(q).astype(np.float32))
        def func_numpy():
            Cinv = tf.linalg.inv(C)
        def func_woodbury():
            Winv = woodbury_inverse_sym_tf(Ainv,W)
        time_numpy = time_method(func_numpy,num=500)
        time_woodbury = time_method(func_woodbury,num=500)
        #time_woodbury2 = time_method(func_woodbury2,num=50)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.3f'%time_numpy)
        print('Woodbury time %0.3f'%time_woodbury)

    print('')
        
def test_woodbury_sldet_tf(time=False):
    print_mtm('woodbury_sldet_tf')
    p = 100
    q = 10
    A = np.eye(p)
    U = rand.randn(p,q)
    C = np.eye(q)
    V = rand.randn(q,p)
    T = A + U.dot(np.dot(C,V))
    sn,ldet = la.slogdet(T)

    Atf = tf.Variable(A.astype(np.float32))
    Utf = tf.Variable(U.astype(np.float32))
    Ctf = tf.Variable(C.astype(np.float32))
    Vtf = tf.Variable(V.astype(np.float32))

    _,ldet_test = woodbury_sldet_tf(Atf,Utf,Ctf,Vtf)
    tolerance(ldet_test.numpy()-ldet,1e-4,'woodbury_sldet_tf ldet check')

    if time:
        p = 5000
        q = 10

        A = np.eye(p)
        U = rand.randn(p,q)
        C = np.eye(q)
        V = rand.randn(q,p)
        Atf = tf.Variable(A.astype(np.float32))
        Utf = tf.Variable(U.astype(np.float32))
        Ctf = tf.Variable(C.astype(np.float32))
        Vtf = tf.Variable(V.astype(np.float32))

        T = A + U.dot(np.dot(C,V))
        Ttf = tf.Variable(T.astype(np.float32))
        def func_numpy():
            sn,ldet = tf.linalg.slogdet(Ttf)
        def func_woodbury():
            _,ldet_test = woodbury_sldet_tf(Atf,Utf,Ctf,Vtf)

        time_numpy = time_method(func_numpy,num=100)
        time_woodbury = time_method(func_woodbury,num=100)
        print('Tensorflow time %0.5f'%time_numpy)
        print('Woodbury time %0.5f'%time_woodbury)
    print('')

def test_woodbury_sldet_sym_tf(time=False):
    print_mtm('woodbury_sldet_sym_tf')
    p = 10
    q = 5
    A = np.diag(np.abs(rand.randn(p)))
    W = rand.randn(p,q)
    C = A + np.dot(W,W.T)

    Atf = tf.Variable(A.astype(np.float32))
    Wtf = tf.Variable(W.astype(np.float32))
    wldet = woodbury_sldet_sym_tf(Atf,Wtf)
    sn,nldet = la.slogdet(C)
    diff = np.abs(wldet-nldet)
    message(diff,1e-5,'woodbury_sldet_sym_tf check')

    if time:
        p = 400
        q = 10
        A = np.diag(10*np.abs(rand.randn(p)))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)

        Ctf = tf.Variable(C.astype(np.float32))
        Atf = tf.Variable(A.astype(np.float32))
        Wtf = tf.Variable(W.astype(np.float32))

        def func_numpy():
            ldet = tf.linalg.logdet(Ctf)
        def func_woodbury():
            ldet2 = woodbury_inverse_sym_tf(Atf,Wtf)
        time_numpy = time_method(func_numpy,num=300)
        time_woodbury = time_method(func_woodbury,num=300)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        p = 40
        q = 2
        A = np.diag(10*np.abs(rand.randn(p)))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)

        Ctf = tf.Variable(C.astype(np.float32))
        Atf = tf.Variable(A.astype(np.float32))
        Wtf = tf.Variable(W.astype(np.float32))

        def func_numpy():
            ldet = tf.linalg.logdet(Ctf)
        def func_woodbury():
            ldet2 = woodbury_inverse_sym_tf(Atf,Wtf)
        time_numpy = time_method(func_numpy,num=300)
        time_woodbury = time_method(func_woodbury,num=300)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        p = 2000
        q = 10
        A = np.diag(10*np.abs(rand.randn(p)))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)

        Ctf = tf.Variable(C.astype(np.float32))
        Atf = tf.Variable(A.astype(np.float32))
        Wtf = tf.Variable(W.astype(np.float32))

        def func_numpy():
            ldet = tf.linalg.logdet(Ctf)
        def func_woodbury():
            ldet2 = woodbury_inverse_sym_tf(Atf,Wtf)
        time_numpy = time_method(func_numpy,num=300)
        time_woodbury = time_method(func_woodbury,num=300)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        p = 2000
        q = 3
        A = np.diag(10*np.abs(rand.randn(p)))
        Ainv = np.diag(1/np.diag(A))
        W = rand.randn(p,q)
        C = A + np.dot(W,W.T)

        Ctf = tf.Variable(C.astype(np.float32))
        Atf = tf.Variable(A.astype(np.float32))
        Wtf = tf.Variable(W.astype(np.float32))

        def func_numpy():
            ldet = tf.linalg.logdet(Ctf)
        def func_woodbury():
            ldet2 = woodbury_inverse_sym_tf(Atf,Wtf)
        time_numpy = time_method(func_numpy,num=300)
        time_woodbury = time_method(func_woodbury,num=300)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Numpy time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

    print('')

def test_woodbury_inverse_block_sym_tf(time=False):
    print_mtm('woodbury_inverse_block_sym_tf')
    N = 11
    p = 10
    q = 5
    A = np.diag(rand.randn(N,p)).astype(np.float32)
    W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
    WT =  np.transpose(W,axes=(0,2,1)).conj()
    AA = tf.Variable(A.astype(np.complex64))
    WW = tf.Variable(W.astype(np.complex64))
    WWT = tf.Variable(WT.astype(np.complex64))
    CC = tf.linalg.diag(AA) + tf.einsum('...ij,...jk->...ik',WW,WWT)

    Cinv = tf.linalg.inv(CC)
    Cinv_woodbury = woodbury_inverse_block_sym_tf(AA,WW)
    dd = Cinv - Cinv_woodbury
    Cdiff = dd.numpy()
    diff = np.mean(np.abs(Cdiff))
    tolerance(diff,1e-3,'woodbury_inverse_block_sym_tf check')

    if time:
        nRep = 300
        N = 100
        p = 40
        q = 3
        A = np.diag(rand.randn(N,p)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        CC = tf.linalg.diag(AA) + tf.einsum('...ij,...jk->...ik',WW,WWT)

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            Cinv_woodbury = woodbury_inverse_block_sym_tf(AA,WW)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)
         
        N = 100
        p = 11
        q = 2
        A = np.diag(rand.randn(N,p)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        CC = tf.linalg.diag(AA) + tf.einsum('...ij,...jk->...ik',WW,WWT)

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            Cinv_woodbury = woodbury_inverse_block_sym_tf(AA,WW)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        N = 200
        p = 150
        q = 3
        A = np.diag(rand.randn(N,p)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        CC = tf.linalg.diag(AA) + tf.einsum('...ij,...jk->...ik',WW,WWT)

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            Cinv_woodbury = woodbury_inverse_block_sym_tf(AA,WW)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)
        
def test_woodbury_inv_isotropic_tf(time=False):
    print_mtm('woodbury_inv_isotropic_tf')
    #'''
    N = 2000
    p = 11
    q = 3
    A = 6*np.abs(rand.rand(N)).astype(np.float32)
    sI = np.zeros((N,p,p))
    for i in range(N):
        sI[i] = A[i]*np.eye(p)

    W = rand.randn(N,p,q) + 1j*rand.randn(N,p,q)
    WT = np.transpose(W,axes=(0,2,1)).conj()
    WW = tf.Variable(W.astype(np.complex64))
    WWT = tf.Variable(WT.astype(np.complex64))
    II = tf.Variable(sI.astype(np.complex64))
    CC = II + tf.einsum('...ij,...jk->...ik',WW,WWT)
    Cinv = tf.linalg.inv(CC)

    AA = tf.Variable(A.astype(np.complex64))
    
    Cinv_woodbury = woodbury_inv_isotropic_tf(AA,WW)

    II_woodbury = tf.einsum('...ij,...jk->...ik',CC,Cinv_woodbury)
    II_inv = tf.einsum('...ij,...jk->...ik',CC,Cinv)

    d1 = tf.reduce_mean(tf.abs(II_woodbury-II))
    d2 = tf.reduce_mean(tf.abs(II_inv-II))

    dd = Cinv - Cinv_woodbury
    Cdiff = dd.numpy()
    diff = np.mean(np.abs(Cdiff))
    tolerance(diff,1e-3,'woodbury_inv_isotropic_tf check')
    print(AA.shape)
    print(WW.shape)
    #'''

    if time:
        nRep = 300
        N = 200
        p = 150
        q = 3
        sI = np.zeros((N,p,p))
        A = 6*np.abs(rand.rand(N)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        II = tf.Variable(sI.astype(np.complex64))
        CC = II + tf.einsum('...ij,...jk->...ik',WW,WWT)

        I1 = np.zeros((N,p,p))
        I2 = np.zeros((N,q,q))
        for i in range(N):
            I1[i] = np.eye(p)
            I2[i] = np.eye(q)
        III1 = tf.constant(I1.astype(np.complex64))
        III2 = tf.constant(I2.astype(np.complex64))

        def func_numpy():
            tf.linalg.inv(CC)
        def func_woodbury():
            #woodbury_inv_isotropic_tf(AA,WW)#,I1=I1,I2=I2)
            woodbury_inv_isotropic_tf(AA,WW,I1=III1,I2=III2)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        N = 200
        p = 150
        q = 10
        A = 6*np.abs(rand.rand(N)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        II = tf.Variable(sI.astype(np.complex64))
        CC = II + tf.einsum('...ij,...jk->...ik',WW,WWT)

        I1 = np.zeros((N,p,p))
        I2 = np.zeros((N,q,q))
        for i in range(N):
            I1[i] = np.eye(p)
            I2[i] = np.eye(q)
        III1 = tf.constant(I1.astype(np.complex64))
        III2 = tf.constant(I2.astype(np.complex64))

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            woodbury_inv_isotropic_tf(AA,WW,I1=III1,I2=III2)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        p = 56
        q = 5
        sI = np.zeros((N,p,p))
        A = 6*np.abs(rand.rand(N)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        II = tf.Variable(sI.astype(np.complex64))
        CC = II + tf.einsum('...ij,...jk->...ik',WW,WWT)

        I1 = np.zeros((N,p,p))
        I2 = np.zeros((N,q,q))
        for i in range(N):
            I1[i] = np.eye(p)
            I2[i] = np.eye(q)
        III1 = tf.constant(I1.astype(np.complex64))
        III2 = tf.constant(I2.astype(np.complex64))

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            woodbury_inv_isotropic_tf(AA,WW,I1=III1,I2=III2)
            #Cinv_woodbury = woodbury_inv_isotropic_tf(AA,WW)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)

        p = 56
        q = 2
        sI = np.zeros((N,p,p))
        A = 6*np.abs(rand.rand(N)).astype(np.float32)
        W = rand.randn(N,p,q) +1j*rand.randn(N,p,q)
        WT =  np.transpose(W,axes=(0,2,1)).conj()
        AA = tf.Variable(A.astype(np.complex64))
        WW = tf.Variable(W.astype(np.complex64))
        WWT = tf.Variable(WT.astype(np.complex64))
        II = tf.Variable(sI.astype(np.complex64))
        CC = II + tf.einsum('...ij,...jk->...ik',WW,WWT)

        I1 = np.zeros((N,p,p))
        I2 = np.zeros((N,q,q))
        for i in range(N):
            I1[i] = np.eye(p)
            I2[i] = np.eye(q)
        III1 = tf.constant(I1.astype(np.complex64))
        III2 = tf.constant(I2.astype(np.complex64))

        def func_numpy():
            Cinv = tf.linalg.inv(CC)
        def func_woodbury():
            woodbury_inv_isotropic_tf(AA,WW,I1=III1,I2=III2)
            #Cinv_woodbury = woodbury_inv_isotropic_tf(AA,WW)
        time_numpy = time_method(func_numpy,num=nRep)
        time_woodbury = time_method(func_woodbury,num=nRep)
        print('p=%d,q=%d'%(int(p),int(q)))
        print('Tensorflow time %0.8f'%time_numpy)
        print('Woodbury time %0.8f'%time_woodbury)
        #'''
##############################################
##############################################
##                                          ##
##  Test declaring block diagonal matrices  ##
##                                          ##
##############################################
##############################################

def test_block_diagonal_square_tf():
    print_mtm('block_diagonal_square_tf')
    aa = np.ones((1,1)).astype(np.float32)
    bb = 3*np.eye(2).astype(np.float32)
    cc = -1*np.ones((3,3)).astype(np.float32)
    A1 = tf.Variable(aa)
    B1 = tf.Variable(bb)
    C1 = tf.Variable(cc)
    outMat = block_diagonal_square_tf([A1,B1,C1])
    sp_Mat = block_diag(aa,bb,cc)
    om_Np = outMat.numpy()
    diff = mat_err(sp_Mat,om_Np)
    message(diff,1e-9,'block_diagonal_square_tf check')
    print('')

def test_block_diagonal_tf():
    print_mtm('block_diagonal_tf')
    a1 = rand.randn(2,3).astype(np.float32)
    a2 = rand.randn(4,2).astype(np.float32)
    a3 = rand.randn(5,5).astype(np.float32)
    A1 = tf.Variable(a1)
    A2 = tf.Variable(a2)
    A3 = tf.Variable(a3)
    sp_Mat = block_diag(a1,a2,a3)
    outMat = block_diagonal_tf([A1,A2,A3])
    om_np = outMat.numpy()
    diff = mat_err(om_np,sp_Mat)
    message(diff,1e-9,'block_diagonal_tf check')
    print('')

if __name__ == "__main__":
    print_ftm('utils_matrix_tf')
    test_subset_matrix_tf()

    '''
    test_woodbury_inverse_tf(time=True)
    test_woodbury_inverse_sym_tf(time=True)

    test_woodbury_sldet_tf(time=True)
    test_woodbury_sldet_sym_tf(time=True)

    test_block_diagonal_square_tf()
    #'''
    #test_block_diagonal_tf()

    #test_woodbury_inverse_block_sym_tf(time=True)
    test_woodbury_inv_isotropic_tf(time=True)

