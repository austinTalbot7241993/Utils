import numpy as np
import sys,os
import pickle
import matplotlib.pyplot as plt
from tensorflow import  keras
import tensorflow as tf
sys.path.append('/home/austin/Utilities/Code/Tensorflow/')
sys.path.append('/home/austin/Utilities/Code/Tensorflow/')
from utils_inception_tf import model_Inception_DL2_tf


def evaluate_mse():


def test_inception_dl2_tf(dataDict,options_dict,standardize=True):
    X = dataDict['X']
    

if __name__ == "__main__":
    myDict = pickle.load(open('NN_Training_data.p','rb'))
    X = myDict['X']
    power_sel = myDict['power_sel']
    coherence_sel = myDict['coherence_Sel']
    granger_sel = myDict['granger_sel']

