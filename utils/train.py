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
from keras.models import load_model

from utils.load_train_val import load_data
from utils.models import *


# load the data
X_train,Y_train,X_val,Y_val = load_data(filename 'gray_scale.npy',kernelfile = 'kernel1.mat', val_percent = 0.04)

# load the model
model = model_CNN_BN(lr = 0.0005,loss_type = 'mean_squared_error',reg_lambda = 0.001)


#create log of the training
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/mse_randominitkernel_model2_batchnorm_lr0.0005', 
                                         histogram_freq=0, write_graph=True, write_images=True)

# reduce lr with respect val loss while training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.0001)
#save the models
model_check = ModelCheckpoint('./models/mse_randominitkernel_model2_batchnorm_lr0.0005/weights.{epoch:02d}-{val_loss:.3f}.hdf5', 
                              monitor='val_loss', verbose=0, save_best_only=False, 
                                save_weights_only=False, mode='auto', period=1)


# load already trained model

model = load_model('./models/mse_randominitkernel_model2_batchnorm_lr0.0005/weights.29-0.01.hdf5')
# train the model

initial_epoch=0
model.fit(X_train, Y_train,
                epochs= 50,
                batch_size=64,
                shuffle=True,
                initial_epoch = initial_epoch,
                validation_data=(X_val, Y_val),
                callbacks=[tbCallBack,reduce_lr,model_check])






