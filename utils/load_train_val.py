# required libraries
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
from scipy.stats import norm,binom
from scipy.io import loadmat
from scipy.special import digamma, gammaln, gamma
from numpy.linalg import inv
import glob
from scipy import ndimage
from PIL import Image
import os, sys
from scipy import signal

def load_data(filename,kernelfile, val_percent):
    """
    function to load the dataset from the npy file
    and return train and val datasets
    """
    # load the dataset
    orig = np.load(filename)

    # resize the dataset for memory efficiency
    orig_resized = np.zeros([orig.shape[0],128,128])
    for i in range(orig.shape[0]):
        orig_resized[i] = cv2.resize(orig[i],(128,128))
    
    # convert the image to uint8 datatype
    orig_resized = orig_resized.astype('uint8')

    # load the kernel to use for convolution
    kernel = loadmat('kernel1.mat')
    kernel = kernel['A']

    # apply convolution on the loaded images to create the input images
    conv = np.zeros_like(orig_resized)
    for i in range(orig_resized.shape[0]):
        #conv[i] = signal.fftconvolve(orig_resized[i],kernel,mode = 'same')
        conv[i] = gaussian_filter(orig_resized[i],sigma=5)

    # create train and val sets

    num_validation = int(orig.shape[0]/25)
    num_training = orig.shape[0] - num_validation

    conv = conv.astype('float16')
    orig_resized = orig_resized.astype('float16')

    X_train, Y_train = conv[:num_training], orig_resized[:num_training]
    X_val, Y_val = conv[num_training:], orig_resized[num_training:]

    print('Xtrain shape: ',X_train.shape)
    print('Ytrain shape: ',Y_train.shape)
    print('Xval shape: ',X_val.shape)
    print('Yval shape: ',Y_val.shape)

    #normalise the dataset
    X_train = X_train / 255.0
    Y_train = Y_train / 255.0
    X_val = X_val / 255.0
    Y_val = Y_val / 255.0

    # reshape the dataset to feed into the keras model

    X_train = X_train.reshape((-1,128,128,1))
    Y_train = Y_train.reshape((-1,128,128,1))
    X_val = X_val.reshape((-1,128,128,1))
    Y_val = Y_val.reshape((-1,128,128,1))

    return X_train,Y_train,X_val,Y_val
