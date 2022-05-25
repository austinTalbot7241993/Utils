'''
This implements an inception-net architecture for EEG/LFP/MEG recordings

Methods
-------
def Inception1D_tf(input_shape=(256,1),options_dict=None):

Attention-Based Network for Weak Labels in Neonatal Seizure Detection

Author : Dmitry Isaev

Contributors:
Dmitry Yu Isaev (Duke University); 
Dmitry Tchapyjnikov (Duke University); 
Michael Cotten (Duke University); 
David Tanaka (Duke University); 
Natalia L Martinez (Duke University); 
Martin A Bertran (Duke University); 
Guillermo Sapiro (Duke University); 
David Carlson (Duke University)

https://github.com/dyisaev/seizure-detection-neonates
/home/austin/Utilities/Code/Tensorflow
'''
import keras.backend as K
from keras.layers import Input, ReLU, Conv1D, GlobalMaxPooling1D 
from keras.layers import AveragePooling1D,MaxPooling1D,Flatten
from keras.layers import TimeDistributed,Dense,BatchNormalization
from keras.layers import Lambda,Softmax,Multiply,GlobalAveragePooling1D
from keras.layers import Activation,ZeroPadding1D,Add,Concatenate,Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.losses import binary_crossentropy,kullback_leibler_divergence
from keras.metrics import binary_accuracy

import tensorflow as tf
import numpy as np

import sys
sys.path.append('/home/austin/Utilities/Code/Miscellaneous')
from utils_misc import fill_dict

def Inception1D_tf(input_shape=(256,1),options_dict=None):
    '''
    Creates an inception block designed for EEGs

    Options
    -------
    nfilt : int,default=16
        Number of filters

    sl1 : int,default=2
        First stride length

    sl2 : int,default=1
        Second stride length

    ks1 : int,default=1
        Smallest kernel size for CNN

    ks2 : int,default=3
        Middle kernel size for CNN

    ks3 : int,default=5
        Largest kernel size for CNN

    ks4 : int,default=7
        First kernel size for CNNs

    act : str,default='relu'
        Activation function for layers
   
    mp1 : int,default=3
        Max pool size
    '''
    default_dict = {'nfilt':16,'sl1':2,'act':'relu','ks1':1,'ks2':3,
                'sl2':1,'ks3':5,'ks4':7,'mp1':3,'nconv':1}
    op = fill_dict(options_dict,default_dict)

    X_input = Input(input_shape)
    X_input_bn = BatchNormalization(axis=-1)(X_input)

    X = X_input_bn
    X = Conv1D(op['nfilt'],op['ks4'],strides=op['sl1'],
                                            activation=op['act'])(X)
    X = MaxPooling1D(op['mp1'],strides=op['sl1'])(X)
    for i in range(op['nconv']):
        X = Conv1D(op['nfilt'],op['ks1'],strides=op['sl2'],activation='relu')(X)
        X = Conv1D(op['nfilt'],op['ks2'],strides=op['sl2'],activation='relu')(X)
        X = MaxPooling1D(op['mp1'],strides=op['sl1'])(X)
    tower_1 = Conv1D(op['nfilt'],op['ks1'],padding='same',
                                            activation=op['act'])(X)
    tower_1 = Conv1D(op['nfilt'],op['ks2'],padding='same',
                                            activation=op['act'])(tower_1)
    tower_2 = Conv1D(op['nfilt'],op['ks1'],padding='same',
                                            activation=op['act'])(X)
    tower_2 = Conv1D(op['nfilt'],op['ks3'],padding='same',
                                            activation=op['act'])(tower_2)
    tower_3 = MaxPooling1D(op['mp1'],strides=op['sl2'],padding='same')(X)
    tower_3 = Conv1D(op['nfilt'],op['ks1'],padding='same',
                                            activation=op['act'])(tower_3)
    X2 = Concatenate(axis = -1)([tower_1,tower_2,tower_3])
    X2 = MaxPooling1D(op['mp1'],strides=op['sl1'])(X2)
    tower_12 = Conv1D(op['nfilt'],op['ks1'],padding='same',
                                            activation=op['act'])(X2)
    tower_12 = Conv1D(op['nfilt'],op['ks2'],padding='same',
                                            activation=op['act'])(tower_12)
    tower_22 = Conv1D(op['nfilt'],op['ks1'],padding='same',
                                            activation=op['act'])(X2)
    tower_22 = Conv1D(op['nfilt'],op['ks3'],padding='same',
                                            activation=op['act'])(tower_22)
    tower_32 = MaxPooling1D(op['mp1'],strides=1,padding='same')(X2)
    tower_32 = Conv1D(op['nfilt'], op['ks1'],padding='same', 
                                            activation=op['act'])(tower_32)
    X3 = Concatenate(axis=-1)([tower_12,tower_22,tower_32])
    global_avp = GlobalAveragePooling1D()(X3)

    model = Model(inputs=[X_input],outputs=[global_avp])
    return model

def model_Inception_VAE_tf(options_dict):
    '''

    '''
    n_channels = options_dict['n_channels']
    n_timesteps = options_dict['n_timesteps']
    n_components = options_dict['n_components']
    wd = options_dict['wd']
    if 'inception_params' not in options_dict.keys():
        print('Hello')
        inp_params = None
    else:
        print('Goodbye')
        inp_params = options_dict['inception_params']

    data_input = Input(shape=(n_channels,n_timesteps),dtype='float32',
                                                        name='input')
    perchan_model = Inception1D_tf((n_timesteps,1),options_dict=inp_params)
    data_input_rs = Lambda(lambda x: K.expand_dims(x, axis=-1), 
                                        name='data_input_rs')(data_input)
    data_processed = TimeDistributed(perchan_model, 
                                    name='data_before_mil')(data_input_rs)
    
    data_attention = TimeDistributed(Dense(32,activation='tanh', 
                kernel_regularizer=l2(wd),use_bias=False))(data_processed)
    data_attention = TimeDistributed(Dense(1,activation=None, 
                kernel_regularizer=l2(wd),use_bias=False))(data_attention)
    data_attention = Lambda(lambda x: K.squeeze(x,-1))(data_attention)
    data_attention = Softmax()(data_attention)
    data_attention = Lambda(lambda x: K.expand_dims(x))(data_attention)
    data_attention = Lambda(lambda x: K.repeat_elements(x, 
                            data_processed.shape[-1],-1),
                            name='att_mil_weights')(data_attention)

    data_weighted = Multiply()([data_processed, data_attention])
    data_sum = GlobalAveragePooling1D()(data_weighted)
    out_dense = Dense(32, activation='relu', 
                                kernel_regularizer=l2(wd))(data_sum)
    out_sq = Dense(2*n_components,activation=None,
                                        name='out_score')(out_dense)
    model = Model(inputs=[data_input], outputs=[out_sq])
    return model

def model_Inception_DL2_tf(options_dict):
    '''

    '''
    finalActivation = options_dict['final_activation']
    n_channels = options_dict['n_channels']
    n_timesteps = options_dict['n_timesteps']
    wd = options_dict['wd']
    if 'inception_params' not in options_dict.keys():
        inp_params = None
    else:
        inp_params = options_dict['inception_params']
    data_input = Input(shape=(n_channels,n_timesteps),dtype='float32',
                                                        name='input')

    perchan_model = Inception1D_tf((n_timesteps,1),options_dict=inp_params)
    
    data_input_rs = Lambda(lambda x: K.expand_dims(x, axis=-1), 
                                        name='data_input_rs')(data_input)
    data_processed = TimeDistributed(perchan_model, 
                                    name='data_before_mil')(data_input_rs)
    
    data_attention = TimeDistributed(Dense(32,activation='tanh', 
                kernel_regularizer=l2(wd),use_bias=False))(data_processed)
    data_attention = TimeDistributed(Dense(1,activation=None, 
                kernel_regularizer=l2(wd),use_bias=False))(data_attention)
    data_attention = Lambda(lambda x: K.squeeze(x,-1))(data_attention)
    data_attention = Softmax()(data_attention)
    data_attention = Lambda(lambda x: K.expand_dims(x))(data_attention)
    data_attention = Lambda(lambda x: K.repeat_elements(x, 
                        data_processed.shape[-1],-1),
                            name='att_mil_weights')(data_attention)

    data_weighted = Multiply()([data_processed, data_attention])
    data_sum = GlobalAveragePooling1D()(data_weighted)
    out_dense = Dense(32, activation='relu', 
                                kernel_regularizer=l2(wd))(data_sum)
    out_sq = Dense(1,activation=finalActivation,name='out_score')(out_dense)

    model = Model(inputs=[data_input], outputs=[out_sq])
    return model
