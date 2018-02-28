###############################################################
# This script reads the data from the input files, creats a   #
#  Convolutional Neural Network and train it.                 #
#                                                             #
# Required files: data_std.pickle, refined_set_dataset.csv    #
#                                                             #
# Written by: Mahmudulla Hassan                               #
# Last Modified: 02/27/2018                                   #
###############################################################

import pickle
import csv
import numpy as np
from tqdm import *
import multiprocessing as mp
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, ZeroPadding3D
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D
from keras import metrics
import random


# Load the data (already normalized). <br>
# Data shape: (Sample size, 165, 285, 250, 8)

print('Loading the data file...')
with open('data_std.pickle', 'rb') as f:
    data = pickle.load(f)

print('Sample size: ', len(data))  


# Read the data file and get all the pdb ids
def read_score():
    pdb_ids = []
    score = {}
    with open('refined_set_dataset.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header
        for row in reader:
            pdb_ids.append(row[1])
            score[row[1]]= row[5]

    return score

# Batch generator for training
def generator(data, batch_size):
    if len(data) < batch_size:
        batch_size = len(data)
    
    score = read_score()
    rand_keys = np.random.choice(list(data.keys()), batch_size)
    data_dim = list(data[list(data.keys())[0]].shape)
    
    while(True):
        x = np.empty([batch_size] + data_dim)
        x[0, :, :, :, :] = data[rand_keys[0]]
        y = np.empty(batch_size)
        y[0] = np.float32(score[rand_keys[0]])

        for i in range(1, len(rand_keys)):
            x[i, :, :, :, :] = data[rand_keys[i]]
            y[i] = score[rand_keys[i]]

        yield x, y

# Model generator
def get_model(summary=False):
    # input_shape = (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)
#     model.add(Convolution3D(filters,
#                             kernel_size,
#                             strides=(1, 1, 1),
#                             padding='valid',
#                             data_format=None,
#                             dilation_rate=(1, 1, 1),
#                             activation=None,
#                             use_bias=True,
#                             kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros',
#                             kernel_regularizer=None,
#                             bias_regularizer=None,
#                             activity_regularizer=None,
#                             kernel_constraint=None,
#                             bias_constraint=None
#                            )
#              )

    model = Sequential()
    
    # 1st layer group
    model.add(Conv3D(filters=64,
                     kernel_size=(5, 5, 5),
                     strides = (2, 2, 2),
                     input_shape=(165, 285, 250, 8),
                     padding='valid',
                     activation='relu',
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           padding='valid'))
    
    # 2nd layer group
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='valid'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           padding='valid'))
    
    # 3rd layer group
    model.add(Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='valid'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(1, 1, 1),
                           padding='valid'))
    
    # 4th layer group
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='valid'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(1, 1, 1),
                           padding='valid'))
    
    # Reduce the number of parameters (!!!)
    model.add(AveragePooling3D(pool_size=(2, 2, 2), 
                               padding='valid'))
                           
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='linear'))
    if summary:
        print(model.summary())
    return model



# Set dynamic memory allocation in a specific gpu
# K.clear_session()
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
K.set_session(K.tf.Session(config=config))

model = get_model(summary=True)
model = multi_gpu_model(model, gpus=4)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[metrics.mse])


batch_size = 5
history = model.fit_generator(generator=generator(data=data, batch_size=batch_size),
                             epochs = 20,
                             steps_per_epoch=len(data)//batch_size,
                             verbose = 1)
