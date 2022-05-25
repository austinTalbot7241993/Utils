import numpy as np

def boxcarAverage(x,num=5):
	return np.convolve(x,np.ones(num)/num,mode='valid')
