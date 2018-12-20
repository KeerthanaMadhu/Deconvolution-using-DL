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
from keras.models import load_data
import keras


# load the data
X_train,Y_train,X_val,Y_val = load_data(filename 'gray_scale.npy',kernelfile = 'kernel1.mat', val_percent = 0.04)

#load saved model
model = load_model('./models/mse_randominitkernel_model2_batchnorm_lr0.0005/weights.29-0.01.hdf5')

# predict on val data

pred = model.predict(X_val)
plt.subplot(1,3,1)
plt.imshow((Y_val[90,:,:,0]*255.0).astype('uint8'))
plt.subplot(1,3,2)
plt.imshow((X_val[90,:,:,0]*255.0).astype('uint8'))
plt.subplot(1,3,3)
plt.imshow((pred[90,:,:,0]*255.0).astype('uint8'))


