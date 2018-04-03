#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:54:51 2017

@author: ofuentes
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
#from keras import initializers
from keras.callbacks import EarlyStopping
#from tf_stance_reg_func_v0 import *
from ofda_weight_init import *
from io_utils import *
from  scipy.stats import pearsonr
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

#print("Reading Data")

dataset = 4

filename = 'affinity_pars_v'+str(dataset)+'.txt'
Data, Target = read_csv_txt(filename ,4631,0)
print(filename,Data.shape)
Data =standardize(Data)
ind = np.arange(Data.shape[0])
version = 3
parfile = 'pars_best_pred_'+str(dataset)+'_'+str(version)+'.npy'
parvec = np.load(parfile)
_, _, dropout, non_zero_frac, _, _, _,lr, l2_penalty,patience = parvec.astype(np.float32)
batch_size, epochs, _, _, h0, h1, h2,_, _,patience = parvec.astype(int)
epochs = 2000

print(batch_size, epochs, dropout, non_zero_frac, h0, h1, h2,lr, l2_penalty,patience)

#batch_size = 1920
#h0, h1, h2 = 300, 200, 150
#dropout=.3

hidden_units = [h0, h1, h2]

max_corr=-1
count=0
repetitions = 100
sum_pred =0
for rep in range(repetitions):
    
    print("Rep:",rep)
    print("Parameters:",batch_size,epochs,dropout,non_zero_frac,hidden_units,lr,l2_penalty,patience)
    
    model = Sequential()   
    model.add(Dropout(.15,input_shape=(Data.shape[1],)))
    for i in range(3):
        model.add(Dense(hidden_units[i], activation='relu',kernel_regularizer=regularizers.l2(l2_penalty)))
        #model.add(Dense(hidden_units[i], activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(Target.shape[1] , activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=lr))
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0., patience=patience, verbose=1, mode='auto')
    earlystop = EarlyStopping(monitor='loss', min_delta=0., patience=patience, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    k=10
    Pred = np.zeros(Target.shape)    
    #ind = np.random.permutation(Data.shape[0])
    #ind = np.arange(Data.shape[0])
    for fold in range(k):
        print("fold:",fold)
        test = ind[ind % k == fold]   
        train = ind[ind % k != fold]
        batch_x, batch_y = get_batch(Data[train,:], Target[train], batch_size)
        W = get_weights_OFDA(batch_x, batch_y, hidden_units, non_zero_frac)
        model.set_weights(W)
        model_info=model.fit(Data[train,:], Target[train],
              batch_size=batch_size,
              validation_data=(Data[test,:], Target[test]),
              epochs=epochs,
              verbose=0,
              callbacks=callbacks_list)
        Pred[test] = model.predict(Data[test,:])   
        plt.plot(model_info.history['loss'])
        plt.title('Loss for fold '+str(fold), fontsize=18, color='black')
        plt.plot(model_info.history['val_loss'])      
        plt.xlabel('Epoch', fontsize=14, color='black')  
        plt.ylabel('MSE', fontsize=14, color='black') 
        plt.show()
    
    #print("Parameters:",batch_size,epochs,dropout,non_zero_frac,hidden_units,l2_penalty,patience )
    err = Target-Pred 
    error2 = err*err
    print("MSE 10fcv",np.mean(error2))
    c, p = pearsonr(Pred,Target)
    print("Pearson correlation:", c[0])
    if(c[0]>max_corr):
        pars = [batch_size,epochs,dropout,non_zero_frac,h0,h1,h2,lr,l2_penalty,patience]
        max_corr=c[0]
        dropout_best = dropout
        lr_best = lr
        l2_penalty_best = l2_penalty
        non_zero_frac_best = non_zero_frac
        patience_best = patience
        h0_best = h0
        h1_best = h1       
        h2_best = h2
        outfile = 'best_pred_'+str(dataset)+'_'+str(count)
        np.save(outfile, Pred)
        np.save('pars_'+outfile, pars)
        np.save('corr_'+outfile, max_corr)
        #model.save_weights('weights')
        count+=1
        plt.plot(Target,Pred,'b.')
        plt.title('Prediction on training data', fontsize=18, color='black')
        plt.xlabel('Affinity', fontsize=14, color='black')  
        plt.ylabel('Prediction', fontsize=14, color='black')
        plt.show()
    
    sum_pred =sum_pred+Pred
    Pred_ens = sum_pred/(rep+1)
    err = Target-Pred_ens
    error2 = err*err
    print("MSE 10fcv ensemble",np.mean(error2))
    c, p = pearsonr(Pred_ens,Target)
    print("Pearson correlation ensemble:", c[0])
    
    dropout = dropout_best * np.random.normal(1,.1) 
    lr = lr_best * np.random.normal(1,.2) 
    l2_penalty = l2_penalty_best * np.random.normal(1,.2) 
    non_zero_frac = non_zero_frac_best * np.random.normal(1,.1) 
    patience = np.absolute(patience_best + np.random.randint(-10,10))
    
    h0 = np.absolute(h0_best + np.random.randint(-20,20))
    h1 = np.absolute(h1_best + np.random.randint(-20,20))
    h2 = np.absolute(h2_best + np.random.randint(-20,20))
    
    hidden_units = [h0, h1, h2]
    
    print("Best Pearson correlation so far:", max_corr)
    print("Best parameters:",pars)

