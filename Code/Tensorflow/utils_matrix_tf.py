'''
This 

Methods
-------
block_diagonal_square_tf
    Constructs a block diagonal matrix from a list of square matrices

block_diagonal_tf

matmul_32_tf(X,Y)
    Matrix multiplies a 3d tensor X by 2 tensor Y

Creation Date 11/19/2021

Version History
'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tensorflow.linalg import LinearOperatorFullMatrix
from tensorflow.linalg import LinearOperatorBlockDiag

##################################
##################################
##                              ##
##  Evaluating quadratic forms  ##
##                              ##
##################################
##################################

def evaluate_quad_vec_tf(X,Sigma):
    pass

def evaluate_quad_mat_tf(X,Sigma):
    pass

def subset_square_matrix_tf(Sigma,idxs):
    '''
    Obtains a subset of a square matrix identical in both directions

    Parameters
    ----------

    Returns
    -------

    '''
    SSub1 = tf.boolean_mask(Sigma,idxs,axis=1)
    SSub = tf.boolean_mask(SSub1,idxs,axis=0)
    return SSub

def subset_matrix_tf(M,idxs1,idxs2):
    SSub1 = tf.boolean_mask(M,idxs1,axis=0)
    SSub = tf.boolean_mask(SSub1,idxs2,axis=1)
    return SSub

def evaluate_invquad_vec_tf(v,Sigma):
    '''
    Evaluates XSigma^{-1}X^T

    Parameters
    ----------
    X : array-like, (N,p)
        Data presumably

    Y : array-like, (p,p)
        Probably covariance matrix

    Returns
    -------
    quad : tf.Float
        The mean evaluated quad product
    '''
    end = tf.linalg.solve(Sigma,X.T)
    prod = tf.multiply(X,tf.transpose(end))
    rs = tf.reduce_sum(prod,axis=1)
    quad = tf.reduce_mean(rs)
    return quad

def evaluate_invquad_mat_tf(X,Sigma):
    '''
    Evaluates XSigma^{-1}X^T

    Parameters
    ----------
    X : array-like, (N,p)
        Data presumably

    Y : array-like, (p,p)
        Probably covariance matrix

    Returns
    -------
    quad : tf.Float
        The mean evaluated quad product
    '''
    end = tf.linalg.solve(Sigma,X.T)
    prod = tf.multiply(X,tf.transpose(end))
    rs = tf.reduce_sum(prod,axis=1)
    quad = tf.reduce_mean(rs)
    return quad


def matmul_32_tf(X,Y):
    '''
    Matrix multiplies a 3d tensor X by 2 tensor Y

    Parameters
    ----------
    X : array-like, (a,b,c)
        3-d tensor

    Y : array-like, (a,c)
        2-d array

    Returns
    -------
    XY : array-like,

    '''
    Y_exp = tf.expand_dims(Y,axis=-1)
    prod = tf.muliply(Y_exp,tf.transpose(X,perm=(0,2,1)))
    XY = tf.reduce_sum(prod,axis=1)
    return XY

#################################################
#################################################
##                                             ##
##  Woodbury matrix inverses and determinants  ##
##                                             ##
#################################################
#################################################

## Needs to be fixed computationally by assuming diagonal A and C
def woodbury_inverse_tf(A,U,C,V):
    '''
    This computes (A+UCV)^{-1}

    Parameters
    ----------

    Returns
    -------

    '''
    A_inv = tf.linalg.inv(A)
    C_inv = tf.linalg.inv(C)
    AiU = tf.matmul(A_inv,U)
    VAi = tf.matmul(V,A_inv)
    VAiU = tf.matmul(V,AiU)
    middle = C_inv + VAiU
    back = tf.linalg.solve(middle,VAi)
    second = tf.matmul(AiU,back)
    inverse = A_inv - second
    return inverse

def woodbury_inverse_sym_tf(Dinv,W):
    '''
    Computes the inverse matrix corresponding to PPCA

    Parameters
    ----------

    Returns
    -------
    '''
    AU = tf.matmul(Dinv,W)
    const = tf.constant(np.eye(W.shape[1]).astype(np.float32))
    middle = const + tf.matmul(tf.transpose(W),AU)
    back = tf.linalg.solve(middle,tf.transpose(AU))
    second = tf.matmul(AU,back)
    inverse = Dinv - second
    return inverse

def woodbury_sldet_tf(A,U,C,V):
    dA = tf.linalg.diag_part(A)
    dC = tf.linalg.diag_part(C)
    Ainv = 1/dA
    Cinv = 1/dC

    ldet_a = tf.reduce_sum(tf.math.log(dA))
    ldet_c = tf.reduce_sum(tf.math.log(dC))
    VA = V*Ainv
    VAU = tf.matmul(VA,U)
    CVAU = tf.linalg.diag(Cinv) + VAU

    s3,ldet_cvau = tf.linalg.slogdet(CVAU)
    det_tot = ldet_c + ldet_a + ldet_cvau
    return s3,det_tot

def woodbury_sldet_sym_tf(A,W,eye=None):
    dA = tf.linalg.diag_part(A)
    Ainv = 1/dA

    ldet_a = tf.reduce_sum(tf.math.log(dA))
    if eye is None:
        eye = np.eye(W.shape[1]).astype(np.float32)
    WT = tf.transpose(W)*Ainv
    IWTW = eye + tf.matmul(WT,W)
    ldet_cvau = tf.linalg.logdet(IWTW)
    log_det = ldet_a + ldet_cvau
    return log_det

def woodbury_inverse_block_tf(A,U,C,V):
    pass

def woodbury_inverse_block_sym_tf(A,W):
    '''
    Computes the inverse matrix when there's a bloc. Used in CSFA 2.0
    Allows for complex values

    Parameters
    ----------

    Returns
    -------
    '''
    WT = tf.math.conj(tf.einsum('...ij->...ji',W))
    Ainv = 1/A
    WTA = tf.einsum('...ij,...j->...ij',WT,Ainv)
    WA = tf.einsum('...ij,...i->...ij',W,Ainv)
    I = tf.constant(np.eye(W.shape[-1]).astype(np.complex64))
    outer = tf.einsum('...ij,...jk->...ik',WTA,W)
    IWTW = I + outer
    IWTWi = tf.linalg.inv(IWTW)
    end = tf.einsum('...ij,...jk->...ik',IWTWi,WTA)
    back = tf.einsum('...ij,...jk->...ik',WA,end)
    inverse = tf.linalg.diag(Ainv) - back
    return inverse

def woodbury_inv_isotropic_tf(A,W,I1=None,I2=None):
    '''
    I1 and I2 shoudld be tf.constants!

    '''
    
    WT = tf.math.conj(tf.einsum('...ij->...ji',W))
    Ainv = 1/A
    if I1 is None:
        I1 = tf.linalg.diag(np.ones(W.shape[:-1]).astype(np.complex64))
    if I2 is None:
        shape = list(W.shape[:-2]) + [W.shape[2]]
        I2 = tf.linalg.diag(np.ones(shape).astype(np.complex64))
    
    AI = tf.einsum('b...,b...->b...',A,I2)
    WTW = tf.einsum('...ij,...jk->...ik',WT,W) 
    M = AI + WTW
    Mi = tf.linalg.inv(M)
    MiW = tf.einsum('...ij,...jk->...ik',Mi,WT)
    WMW = tf.einsum('...ij,...jk->...ik',W,MiW)
    IWMW = I1 - WMW
    AI2 = tf.expand_dims(Ainv,axis=-1)
    AI3 = tf.expand_dims(AI2,axis=-1)
    inverse= tf.multiply(AI3,IWMW)
    return inverse


def woodbury_ldet_isotropic_tf(A,W,I1=None):
    pass

def woodbury_sldet_block_tf(A,U,C,V):
    pass

def woodbury_sldet_block_sym_tf(A,W):
    pass

###############################
###############################
##                           ##
##  Creating block matrices  ##
##                           ##
###############################
###############################

def build_blockMatrix_tf(A,B,C,D):
    '''
    Creates block matrix

    M = |A B|
        |C D|

    Paramaeters
    -----------
    A : tf.array
        Top left matrix

    B : tf.array
        Top right matrix

    C : tf.array
        Bottom left matrix

    D : tf.array
        Bottom right matrix

    Returns
    -------
    M : tf.array
        The output matrix
    '''
    top_row = tf.concat([A,B],1)
    bottom_row = tf.concat([C,D],1)
    M = tf.concat([top_row,bottom_row],0)
    return M

def block_diagonal_square_tf(matrices):
    '''
    Constructs a block diagonal matrix from a list of square matrices

    Parameters
    ----------
    matrices : list
        List of square matrices in tensorflow

    Returns 
    -------
    block_matrix : tf.Tensor,shape=(p,p)
        The matrices in block diagonal form
    '''
    linop_blocks = [LinearOperatorFullMatrix(block) for block in matrices]
    linop_block_diagonal = LinearOperatorBlockDiag(linop_blocks)
    block_matrix = linop_block_diagonal.to_dense()
    return block_matrix

# Does not work with no time to debug!
def block_diagonal_tf(matrices, dtype=tf.float32):
    """
    Constructs block-diagonal matrices from a list of batched 2D tensors.

    Parameters
    ----------
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must 
        match this dtype

    Returns
    -------
        A matrix with the input matrices stacked along its main diagonal, 
        having shape [..., \sum_i N_i, \sum_i M_i].
    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]

    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
            [(row_before_length, row_after_length)]],
            axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

