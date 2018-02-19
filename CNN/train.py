###############################################################
# This script reads the data from the input files, creats a   #
#  Convolutional Neural Network and train it.                 #
#                                                             #
# Required files: data_std.pickle, refined_set_dataset.csv    #
#                                                             #
# Written by: Mahmudulla Hassan                               #
# Last Modified: 02/18/2018                                   #
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

def generator(data, batch_size):
    score = read_score()
    rand_keys = np.random.choice(list(data.keys()), batch_size)
    x = np.empty([batch_size, 165, 285, 250, 8])
    x[0, :, :, :, :] = data[rand_keys[0]]
    y = np.empty(batch_size)
    y[0] = np.float32(score[rand_keys[0]])
    
    while(True):
        for i in range(1, len(rand_keys)):
            x[i, :, :, :, :] = data[rand_keys[i]]
            y[i] = score[rand_keys[i]]
    
        yield (x, y)

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
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

    model.add(Conv3D(filters=64,
                     kernel_size=(5, 5, 5),
                     input_shape=(165, 285, 250, 8),
                     padding='same',
                     dilation_rate=(1, 1, 1),
                     activation='relu',
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), 
                           strides=(2, 2, 2), 
                           padding='valid'))
    
    # 2nd layer group
#    model.add(Conv3D(filters=128,
#                     kernel_size=(3, 3, 3),
#                     activation='relu',
#                     padding='same',
#                     dilation_rate=(1, 1, 1)))
    
#    model.add(MaxPooling3D(pool_size=(2, 2, 2), 
#                           strides=(2, 2, 2), 
#                           padding='valid'))
#     # 3rd layer group
#     model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
#                             border_mode='same', name='conv3a',
#                             subsample=(1, 1, 1)))
#     model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
#                             border_mode='same', name='conv3b',
#                             subsample=(1, 1, 1)))
#     model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), 
#                            border_mode='valid', name='pool3'))
#     # 4th layer group
#     model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
#                             border_mode='same', name='conv4a',
#                             subsample=(1, 1, 1)))
#     model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
#                             border_mode='same', name='conv4b',
#                             subsample=(1, 1, 1)))
#     model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), 
#                            border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(filters=512,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     dilation_rate=(1, 1, 1)))
    
#    model.add(Conv3D(filters=512,
#                     kernel_size=(3, 3, 3),
#                     activation='relu', 
#                     padding='same',
#                     dilation_rate=(1, 1, 1)))
    
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), 
                           strides=(2, 2, 2), 
                           padding='valid'))
    model.add(Flatten())
    # FC layers group
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='linear'))
    if summary:
        print(model.summary())
    return model

# Create the model
model = get_model()
# Print the model summary
model.summary()
# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[metrics.mse])

# history = model.fit(x=, y=, batch_size=5, epochs=10, validation_split=0.2, verbose=1 , use_multiprocessing=True, )

# Train the model
history = model.fit_generator(generator=generator(data=data, batch_size=5),
                              epochs=5,
                              steps_per_epoch=6,
                              validation_data=generator(data=data, batch_size=1),
                              validation_steps=2,
                              verbose=1, 
                              workers=mp.cpu_count())#,
#                              use_multiprocessing=True)

