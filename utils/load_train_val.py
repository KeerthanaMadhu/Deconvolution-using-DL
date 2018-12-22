# required libraries
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
from scipy.stats import norm,binom
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import glob
from scipy import ndimage
from PIL import Image
import os, sys
from scipy import signal
import cv2

def load_dataset(filename,kernelfile, val_percent,resize_outs = False,add_noise = False):
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
    print('images loaded')
    # load the kernel to use for convolution
    kernel = loadmat('kernel1.mat')
    kernel = kernel['A']

    # apply convolution on the loaded images to create the input images
    conv = np.zeros_like(orig_resized)
    for i in range(orig_resized.shape[0]):
        #conv[i] = signal.fftconvolve(orig_resized[i],kernel,mode = 'same')
        conv[i] = gaussian_filter(orig_resized[i],sigma=1)
        if add_noise:
            conv[i] = sp_noise(conv[i],0.05)
    print('convolution done')
    # create train and val sets

    num_validation = int(orig.shape[0]/25)
    num_training = orig.shape[0] - num_validation

    #conv = conv.astype('float16')
    #orig_resized = orig_resized.astype('float16')

    X_train, Y_train = conv[:num_training], orig_resized[:num_training]
    X_val, Y_val = conv[num_training:], orig_resized[num_training:]
    
    
    if resize_outs:
        Y_val_resized = np.zeros([Y_val.shape[0],118,118])
        Y_train_resized = np.zeros([Y_train.shape[0],118,118])
        for i in range(Y_train.shape[0]):
            Y_train_resized[i] = cv2.resize(Y_train[i],(118,118))
        for i in range(Y_val.shape[0]):
            Y_val_resized[i] = cv2.resize(Y_val[i],(118,118))
        Y_val = Y_val_resized
        Y_train = Y_train_resized
        
    print('Xtrain shape: ',X_train.shape)
    print('Ytrain shape: ',Y_train.shape)
    print('Xval shape: ',X_val.shape)
    print('Yval shape: ',Y_val.shape)
    
    #X_train = X_train.astype('float16')
    #Y_train = Y_train.astype('float16')
    #X_val = X_val.astype('float16')
    #Y_val = Y_val.astype('float16')
    
    #normalise the dataset
    X_train = X_train.astype('float16') / 255.0
    Y_train = Y_train.astype('float16') / 255.0
    X_val = X_val.astype('float16') / 255.0
    Y_val = Y_val.astype('float16') / 255.0

    # reshape the dataset to feed into the keras model
    
    X_train = X_train.reshape((-1,128,128,1))
    Y_train = Y_train.reshape((-1,Y_train.shape[1],Y_train.shape[2],1))
    X_val = X_val.reshape((-1,128,128,1))
    Y_val = Y_val.reshape((-1,Y_val.shape[1],Y_val.shape[2],1))
    
    return X_train,Y_train,X_val,Y_val

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
