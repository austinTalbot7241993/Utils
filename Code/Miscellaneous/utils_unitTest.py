'''
This implements methods that are useful for unit tests

Methods
-------
message(diff,thresh,message)
    Original method equivalent to tolerence continued for old unit tests

tolerance(diff,thresh,message)
    Checks if diff is close to zero with some tolerance. If failed raises
    error

greater_than(val1,val2,message)
    Sees if a value is greater than specified value

print_otm(objectName)
    Prints a message indicating which object we are testing

print_mtm(methodName)
    Prints a message indicating which object we are testing

print_ftm(objectName)
    Prints a message indicating which object we are testing

time_method(method,num=1000)
    This method times a function call. Useful for evaluating the 
    computational benefits of some methods.
'''

# Author: Austin Talbot <austin.talbot1993@gmail.com>
import numpy as np
import time
from tqdm import trange

def message(diff,thresh,message):
    '''
    Original method equivalent to tolerence continued for old unit tests

    '''
    if np.abs(diff) < thresh:
        mystr = message+' passed: %0.8f < %0.8f'%(np.abs(diff),
                                                thresh)
        print(mystr)
    else:
        mystr = message+' failed: %0.8f > %0.8f'%(np.abs(diff),
                                                thresh)
        raise ValueError(mystr)

def tolerance(diff,thresh,message):
    '''
    Checks if diff is close to zero with some tolerance. If failed raises
    error

    Parameters
    ----------
    diff : float
        Value to check

    thresh : float
        How close we need o be to 0

    message : str
        What message to print

    Returns
    -------
    None
    '''
    if np.abs(diff) < thresh:
        mystr = message+' passed: %0.8f < %0.8f'%(np.abs(diff),
                                                thresh)
        print(mystr)
    else:
        mystr = message+' failed: %0.8f > %0.8f'%(np.abs(diff),
                                                thresh)
        raise ValueError(mystr)

def greater_than(val1,val2,message):
    '''
    Sees if a value is greater than specified value

    Parameters
    ----------
    val1 : float
        Value to check

    val2 : float
        Baseline acceptable value

    message : str
        What message to print


    Returns
    -------
    None
    '''
    if val1 > val2:
        mystr = message + ' passed: %0.8f > %0.8f'%(val1,val2)
        print(mystr)
    else:
        mystr = message + ' failed: %0.8f < %0.8f'%(val1,val2)
        raise ValueError(mystr)

def print_otm(objectName):
    '''
    Prints a message indicating which object we are testing

    Parameters
    ----------
    objectName : str
        The object
    '''
    str1 = '###  Testing methods associated with ' + objectName + ' object  ###'
    tChars = int(len(str1) - 6)
    str2 = '###' + ' '*tChars + '###'
    str3 = '#'*len(str1)
    print(str3)
    print(str3)
    print(str2)
    print(str1)
    print(str2)
    print(str3)
    print(str3)
    print('')
    print('')
    print('')

def print_mtm(methodName):
    '''
    Prints a message indicating which object we are testing

    Parameters
    ----------
    methodName : str
        The method we are testing
    '''
    str1 = '# Testing ' + methodName + ' #'
    str2 = '#'*len(str1)
    print(str2)
    print(str1)
    print(str2)

def print_ftm(objectName):
    '''
    Prints a message indicating which object we are testing

    Parameters
    ----------
    objectName : str
        The object
    '''
    str1 ='###  Testing methods associated with ' + objectName + ' file ###'
    tChars = int(len(str1) - 6)
    str2 = '###' + ' '*tChars + '###'
    str3 = '#'*len(str1)
    print(str3)
    print(str3)
    print(str2)
    print(str1)
    print(str2)
    print(str3)
    print(str3)
    print('')
    print('')
    print('')

def time_method(method,num=1000,tr=True):
    '''
    This method times a function call. Useful for evaluating the 
    computational benefits of some methods.

    Parameters
    ----------
    method : function

    num : int
        Number of iterations

    Returns
    -------
    average_time : float
        The time to run each method requires
    '''
    startTime = time.time()
    if tr:
        for i in trange(int(num)):
            method()
        endTime = time.time()
        average_time = (endTime - startTime)/num
    else:
        for i in range(int(num)):
            method()
        endTime = time.time()
        average_time = (endTime - startTime)/num
    return average_time
