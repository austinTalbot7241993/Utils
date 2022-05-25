import numpy as np
import numpy.random as rand
import tensorflow as tf
from tensorflow import keras
import numpy.linalg as la


def simple_batcher(batchSize,N):
    idx = rand.choice(N,size=batchSize,replace=False)

def simple_batcher_X(batchSize,X):
    N = X.shape[0]
    idx = rand.choice(N,size=batchSize,replace=False)
    X_batch = X[idx]
    return X_batch

def simple_batcher_XY(batchSize,X,Y):
    N = X.shape[0]
    idx = rand.choice(N,size=batchSize,replace=False)
    X_batch = X[idx]
    Y_batch = Y[idx]
    return X_batch,Y_batch

def simple_batcher_csfa_XY(batchSize,X,Y):
    N = Y.shape[0]
    idx = rand.choice(N,size=batchSize,replace=False)
    X_batch = X[:,:,idx]
    Y_batch = Y[idx]
    return X_batch,Y_batch

def simple_batcher_csfa_XY(batchSize,X,Y):
    N = Y.shape[0]
    idx = rand.choice(N,size=batchSize,replace=False)
    X_batch = X[:,:,idx]
    Y_batch = Y[idx]
    return X_batch,Y_batch

def simple_batcher_csfa_XYZ(batchSize,X,Y,Z):
    N = Y.shape[0]
    idx = rand.choice(N,size=batchSize,replace=False)
    X_batch = X[:,:,idx]
    Y_batch = Y[idx]
    Z_batch = Z[idx]
    return X_batch,Y_batch,Z_batch



