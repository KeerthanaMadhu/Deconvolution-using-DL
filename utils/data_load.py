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

# first dataset load

class Image_data(object):
    """
    all functions related to data loading and creating convoluted images

    """

    def __init__(self, npyfilename, pathstodir, kernel_file_name):

        """
          npyfilename : names of .npy files to load
          pathtodir : path to additional images to add to the dataset
          kernel_size : kernel size of the filter to get convoluted images
          kernel_type : gaussian, random,
         """
        images_gray1 = np.load(npyfilename)
        print('loaded npy file')
        #VOC load
        images_PASCAL = []
        for path in pathstodir:
            dirs = os.listdir( path )
            for i in dirs:
                image = scipy.ndimage.imread(path+i,mode = 'L')
                image = scipy.misc.imresize(image,(224,224))
                images_PASCAL.append(image)
            print('loaded VOC data in'+path)   
        images_PASCAL = np.asarray(images_PASCAL)
        self.original_images = np.vstack((images_gray1,images_PASCAL))
        kernel = loadmat(kernel_file_name)
        self.kernel = kernel['A']

        self.total_image_num = self.original_images.shape[0]
        print('original images loaded')
    
    def compute_image_conv(self):
        conv_images = []
        for i in range(self.total_image_num):
            conv_images.append(signal.fftconvolve(self.original_images[i],self.kernel,mode = 'same'))
        self.conv_images = np.asarray(conv_images)
        print('conv images created')

    def normalise(self):
        self.original_images = self.original_images/255
        self.conv_images = self.conv_images
            
    






