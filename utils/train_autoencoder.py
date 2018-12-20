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

# load the dataset
orig = np.load('gray_scale.npy')

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

# model definition 
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
adam = optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy')

#print the model summary
autoencoder.summary()

# train the model
autoencoder.fit(X_train, Y_train,
                epochs= 50,
                batch_size=52,
                shuffle=True,
                validation_data=(X_val, Y_val),
                callbacks=[tbCallBack])




