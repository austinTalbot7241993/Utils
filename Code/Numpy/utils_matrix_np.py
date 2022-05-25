'''
This implements some general-puprose useful matrix methods such as the
woodbury matrix formula for inverses and determinants, as well as evaluating
quadratic forms

Methods :
def evalsEvec_np(Sigma):
    Returns sorted eigenvalues and eigenvectors

subset_square_matrix_np(Sigma,idxs)
    This returns a symmetric subset of a square matrix

evaluate_quad_np(X,Sigma)
    Evaluates \sum{i=1}^N X.T Sigma X

evaluate_invquad_np(X,Sigma)
    Evaluates \sum{i=1}^N X.T Sigma^{-1} X

woodbury_sldet_np(A,U,C,V)
    This computes the log determinant of (A + UCV)
    This makes two major assumptions based on common applications. A and C
    are positive definite and diagonal (no negative entries). Apparently 
    solve is not intelligent enough to recognize this.

woodbury_inverse_np(A,U,C,V)
    This computes the Woodbury inverse given matrices A,U,C,V. See
    wikipedia (A + UCV)^{-1}

woodbury_inverse_sym_np(Dinv,W,eye=None)
    Computes the inverse matrix corresponding to PPCA (C=I, U=VT and Ainv).
    Only substantially faster when eye is explicitly passed in  or q << p

def woodbury_sldet_sym_np(A,W,eye=None)
    Computes the log determinant of C = A + WWT where A is a diagonal matrix

Creation Date 11/09/2021

Version History
    1.1 12/24/2021 - Made a number of methods faster at cost of more 
                     assumptions. Added determinant computation
'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
import numpy as np
import numpy.linalg as la
import numpy.random as rand

def evalsEvec_np(Sigma):
    '''
    Returns sorted eigenvalues and eigenvectors

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        Covariance matrix (presumably)

    Returns
    -------
    eigenValues :

    eigenVectors :
        
    '''
    eigenValues,eigenVectors = la.eig(Sigma)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues,eigenVectors

def evaluate_quad_np(X,Sigma):
    '''
    Evaluates \sum{i=1}^N X.T Sigma X

    Parameters
    ----------
    X : np.array-like,(N,p)
        Data

    Sigma : np.array
        Presumably covariance matrix

    Returns
    -------
    quad : float
        Average value
    '''
    SX = np.dot(X,Sigma)
    XSX = X*SX
    row_avg = np.sum(XSX,axis=1)
    quad = np.mean(row_avg)
    return quad

def evaluate_invquad_vec_np(x,Sigma):
    x = np.squeeze(x)
    XS = la.solve(Sigma,x)
    out = np.dot(x,XS)
    return out

def evaluate_invquad_np(X,Sigma):
    '''
    Evaluates \sum{i=1}^N X.T Sigma^{-1} X

    Parameters
    ----------
    X : np.array-like,(N,p)
        Data

    Sigma : np.array
        Presumably covariance matrix

    Returns
    -------
    quad : float
        Average value
    '''
    SiX = la.solve(Sigma,X)
    XSiX = X*SiX
    row_avg = np.sum(XSiX,axis=1)
    prod = np.mean(row_avg)
    return quad

def subset_square_matrix_np(Sigma,idxs):
    '''
    This returns a symmetric subset of a square matrix

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        Covariance matrix (presumably)

    idxs : np.array-like,(p,)
        Binary vector, 1 means select
    Returns
    -------
    Sigma_sub : np.array-like,(sum(idxs),sum(idxs))
        The subset of matrix
    '''
    Sigma_sub = Sigma[np.ix_(idxs==1,idxs==1)]
    return Sigma_sub

def woodbury_inverse_np(A,U,C,V):
    '''
    This computes the Woodbury inverse given matrices A,U,C,V. See
    wikipedia (A + UCV)^{-1}

    Parameters
    ----------
    A : np.array-like,(p,p)
        First matrix

    U : np.array-like,(p,q)
        Second matrix

    C : np.array-like,(q,q)
        Third matrix

    V : np.array-like,(q,p)
        Fourth matrix

    Returns
    -------
    inverse : np.array,shape=(p,p)
        Inverse of matrix
    '''
    A_inv = np.diag(1/np.diag(A))
    C_inv = np.diag(1/np.diag(C))
    AiU = np.dot(A_inv,U)
    VAi = np.dot(V,A_inv)
    VAiU = np.dot(V,AiU)
    middle = C_inv + VAiU
    back = la.solve(middle,VAi)
    second = np.dot(AiU,back)
    inverse = A_inv - second
    return inverse

def woodbury_sldet_np(A,U,C,V):
    '''
    This computes the log determinant of (A + UCV)
    This makes two major assumptions based on common applications. A and C
    are positive definite and diagonal (no negative entries). Apparently 
    solve is not intelligent enough to recognize this.

    Parameters
    ----------

    Returns
    -------
    
    '''
    ldet_a = np.sum(np.log(np.diag(A)))
    ldet_c = np.sum(np.log(np.diag(C)))
    Ainv = np.diag(1/np.diag(A))
    AU = np.dot(Ainv,U)
    VAU = np.dot(V,AU)
    inv_c = np.diag(1/np.diag(C))
    CVAU = inv_c + VAU
    s3,ldet_cvau = la.slogdet(CVAU) 
    sign = s3
    det_tot = ldet_c + ldet_a + ldet_cvau
    return sign,det_tot

def woodbury_inverse_sym_np(Dinv,W,eye=None):
    '''
    Computes the inverse matrix corresponding to PPCA (C=I, U=VT and Ainv).
    Only substantially faster when eye is explicitly passed in  or q << p

    Parameters
    ----------
    Dinv : np.array(p,p)
        Probably diagonal, matrix pre-inverted
    
    W : np.array(p,q)
        Loadings
    
    eye : np.array(q,q)
        Optional preinitialized array

    Returns
    -------
    inverse : np.array(p,p)
        Inverted matrix
    '''
    AU = np.dot(Dinv,W)
    if eye is None:
        middle = np.eye(W.shape[1]) + np.dot(W.T,AU)
    else:
        middle = eye + np.dot(W.T,AU)
    back = la.solve(middle,AU.T)
    second = np.dot(AU,back)
    inverse = Dinv - second
    return inverse

def woodbury_sldet_sym_np(A,W,eye=None):
    '''
    Computes the log determinant of C = A + WWT where A is a diagonal matrix

    Parameters
    ----------
    A : np.array-like(p,p)
        Diagonal array

    W : np.array-like(p,q)
        Factors
    
    eye : optional,np.array-like(q,q)
        Optional preinitialized matrix to reduce computation power

    Returns
    -------
    log_det : float
        The log determinant
    '''
    dA = np.diag(A) 
    ldet_a = np.sum(np.log(dA))
    if eye is None:
        eye = np.eye(W.shape[1])
    Ainv = 1/dA 
    WT = W.T*Ainv
    IWTW = eye + np.dot(WT,W)
    s3,ldet_cvau = la.slogdet(IWTW) 
    log_det = ldet_a + ldet_cvau
    return log_det


def quadratic_form_np(Lambda,mu,Sigma):
    '''

    '''
    mu = np.squeeze(mu)
    LS = np.dot(Lambda,Sigma)
    term1 = np.trace(LS)
    term2 = np.dot(mu,np.dot(Lambda,mu))
    quad_form = term1 + term2
    return quad_form

def quadratic_form_batch_np(Lambda,Mu,Sigma):
    '''

    '''
    LS = np.dot(Lambda,Sigma)
    term1 = np.trace(LS)
    term2 = evaluate_quad_np(Mu,Lambda)
    quad_form = term1 + term2
    return quad_form
    




