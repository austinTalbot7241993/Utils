'''

Author : Austin Talbot <austin.talbot1993@gmail.com>

Creation Date: 11/04/2021

Version History

Methods 

cosineSimilarity
    
'''
import numpy as np
import numpy.linalg as la

def cosine_similarity_np(vec1,vec2):
    v1 = np.squeeze(vec1)
    v2 = np.squeeze(vec2)
    num = np.dot(v1,v2)
    denom = la.norm(v1)*la.norm(v2)
    return num/denom
