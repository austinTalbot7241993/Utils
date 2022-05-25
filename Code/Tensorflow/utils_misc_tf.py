'''

Methods:

def copy_model(model_origin,model_target):
    Copies a particular sequential model to a new model

return_optimizer_tf(trainingMethod,learningRate,options=None)
    Creates a Keras optimizer with specific parameters

limitGPU(gpuMem)
    Limits the GPU memory to a certain amount

Creator: Austin Talbot <austin.talbot1993@gmail.com>

'''
import numpy as np
import pickle
import numpy.random as rand
from sklearn import decomposition as dp
from sklearn import linear_model as lm
from tensorflow import keras
import tensorflow as tf

def copy_model(model_origin,model_target):
    '''
    Copies a particular sequential model to a new model

    Parameters
    ----------
    model_origin : tensorflow model
        The model with learned weights

    model_target : tensorflow model
        Model with unlearned weights

    Returns
    -------
    model_target : tensorflow model
        The model with identical weights to model_origin
    '''
    for l_orig,l_targ in zip(model_origin.layers,model_target.layers):
        l_targ.set_weights(l_orig.get_weights())
    return model_target

def limitGPU(gpuMem):
    '''
    Limits the GPU memory to a certain amount

    Parameters
    ----------
    gpuMem : int
        MB of memory to allocate
    '''
    gu = int(gpuMem)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gu)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus),"Physical GPUs",len(logical_gpus),"Logical GPU")
        except RuntimeError as e:
            print(e)

def return_optimizer_tf(trainingMethod,learningRate,options=None):
    '''
    Creates a Keras optimizer with specific parameters

    Parameters
    ----------
    trainingMethod : str \in {'Nadam','Adam'}
        The SGD method

    learningRate : float
        The learning rate of optimization

    options : dict
        Misc options, currently unused

    Returns
    -------
    optimizer : keras optimizer
    '''
    if trainingMethod == 'Nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learningRate)
    elif trainingMethod == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    elif trainingMethod == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    else:
        print('Unrecognized learning strategy %s',trainingMethod)
    return optimizer

def return_optimizer_adaptive_tf(trainingMethod,learningRate,options):
    '''
    Creates a Keras optimizer with specific parameters

    Parameters
    ----------
    learningRate : float
        The learning rate of optimization

    options : dict
        Misc scheduling options

    Returns
    -------
    optimizer : keras optimizer
    '''
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=learningRate,
                                decay_steps=options['steps'],
                                decay_rate=options['rate'],
                                staircase=options['staircase'])
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    return optimizer
    

