import numpy as np
import numpy.random as rand
import numpy.linalg as la
from scipy import stats as st
import tensorflow as tf
import pickle

import sys,os
sys.path.append('../../Code/Miscellaneous')
sys.path.append('../../Code/Tensorflow')
from utils_unitTest import tolerance,greater_than
from utils_inception_tf import *

rand.seed(1993)

def test_Inception1D_tf():
    print('##########################')
    print('# Testing Inception1D_tf #')
    print('##########################')
    model = Inception1D_tf()
    print(model.summary())

def test_model_Inception_VAE_tf():
    print('##################################')
    print('# Testing model_Inception_VAE_tf #')
    print('##################################')
    options_dict = {}
    options_dict['n_channels'] = 11
    options_dict['n_timesteps'] = 500
    options_dict['n_components'] = 10
    options_dict['wd'] = 0.005
    model = model_Inception_VAE_tf(options_dict)
    print(model.summary())

def test_model_Inception_DL2_tf():
    print('##################################')
    print('# Testing model_Inception_DL2_tf #')
    print('##################################')
    options_dict = {}
    options_dict['n_channels'] = 11
    options_dict['n_timesteps'] = 500
    options_dict['wd'] = 0.005
    options_dict['final_activation'] = 'sigmoid'
    model = model_Inception_DL2_tf(options_dict)
    print(model.summary())


if __name__ == "__main__":
    test_Inception1D_tf()
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    test_model_Inception_DL2_tf()
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>')
    test_model_Inception_VAE_tf()











