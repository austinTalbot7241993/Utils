'''
This implements methods related to the gaussian distribution in numpy

Methods:

Author : Austin Talbot <austin.talbot1993@gmail.com>

Version History:

'''
import numpy as np
import numpy.linalg as la

def convert_float32_np(list_arrays):
    n_items = len(list_arrays)
    for i in range(n_items):
        list_arrays[i] = list_arrays[i].astype(np.float32)
    return list_arrays

def convert_complex64_np(list_arrays):
    n_items = len(list_arrays)
    for i in range(n_items):
        list_arrays[i] = list_array[i].astype(np.complex64)
    return list_arrays
    
def outer_np(w1,w2):
    s1 = w1.shape[0]
    s2 = w1.shape[1]
    s3 = w2.shape[2]
    out = np.zeros((s1,s2,s3)).astype(np.complex64)
    for i in range(s1):
        out[i] = np.dot(w1[i],w2[i])
    return out 

def outer4_np(w1,w2):
    s1 = w1.shape[0]
    s2 = w1.shape[1]
    s3 = w1.shape[2]
    s4 = w2.shape[3]
    out = np.zeros((s1,s2,s3,s4)).astype(np.complex64)
    for i in range(s1):
        for j in range(s2):
            out[i,j] = np.dot(w1[i,j,:,:],w2[i,j,:,:])
    return out         
