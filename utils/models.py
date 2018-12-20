import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import signal
from scipy.ndimage import gaussian_filter
from keras import regularizers
import cv2

from keras.layers import Conv2D, BatchNormalization, Activation , MaxPooling2D, UpSampling2D 
from keras.models import Model, Input
from keras.optimizers import Adam
from keras import optimizers
import keras.backend as K
import keras


def model_autoencoder(lr = 0.001,loss_type = 'mean_squared_error',reg_lambda = 0.001):
    input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    adam = optimizers.Adam(lr=lr)
    autoencoder.compile(optimizer=adam, loss=loss_type)

    #print the model summary
    autoencoder.summary()

    return autoencoder

def model_autoencoder_bigger(lr = 0.001,loss_type = 'mean_squared_error',reg_lambda = 0.001):

    input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*1)),
            kernel_regularizer=regularizers.l2(reg_lambda))(input_img)
    x = MaxPooling2D((2, 2), padding='valid')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    x = MaxPooling2D((2, 2), padding='valid')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    encoded = MaxPooling2D((2, 2), padding='valid')(x)



    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(encoded)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
                 kernel_regularizer=regularizers.l2(reg_lambda))(x)

    autoencoder = Model(input_img, decoded)
    adam = optimizers.Adam(lr=lr)
    autoencoder.compile(optimizer=adam, loss=loss_type)
    autoencoder.summary()

    return autoencoder

def model_conv_BN(lr = 0.001,loss_type = 'mean_squared_error',reg_lambda = 0.001):
    input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (7, 7), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*1)),
            kernel_regularizer=regularizers.l2(reg_lambda))(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
           kernel_regularizer=regularizers.l2(reg_lambda))(x)
    out = Conv2D(1, (3, 3), activation='sigmoid', padding='same',kernel_initializer=initializers.RandomNormal(stddev = 2/(3*3*16)),
                 kernel_regularizer=regularizers.l2(reg_lambda))(x)

    conv_BN = Model(input_img, out)
    conv_BN = optimizers.Adam(lr=lr)
    conv_BN.compile(optimizer=adam, loss=loss_type)

    conv_BN.summary()

    return conv_BN
    




