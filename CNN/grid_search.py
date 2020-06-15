import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import callbacks
from keras.metrics import FalseNegatives
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from utils import *
import random
from sklearn.metrics import roc_auc_score, auc
import json


dataDir = '/data/disappearingTracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn_gs/'
weightsDir = workDir + 'weights/cnn_gs/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

###########config parameters##################
fname = 'images_DYJets50_tanh_0p5.pkl'
pos_class = [1]
neg_class = [0,2]
batch_size = 128
max_epochs = 1
patience_count = 20
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
n_kfolds = 5
#############################################

# extract data and classes
df = pd.read_pickle(dataDir+fname)
df_recofail = df.loc[df['deltaRToClosestElectron']>0.15]
x = df_recofail.iloc[:,4:].to_numpy()
x = np.reshape(x, [x.shape[0],40,40,4])
x = x[:,:,:,[0,2,3]]
y = df_recofail['type'].to_numpy()
for i,label in enumerate(y):
    if label in pos_class: y[i] = 1
    if label in neg_class: y[i] = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


def build_model(layers=1,filters=64,opt='adadelta',kernels=(1,1)):
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    for i in range(layers-1):
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

    return model


def kfold_network(X, y, kfolds, layers, filters, opt):
    
    numSplits = 0
    
    model = build_model(layers, filters, opt)
    model.save_weights(weightsDir + 'model_init.h5')

    #early stopping
    patienceCount = patience_count
    cbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=patience_count),
                 callbacks.ModelCheckpoint(filepath=weightsDir+str(numSplits)+'_nlayers'+str(layers)+'_nfilters'+str(filters)+'_opt'+opt+'.h5', monitor='val_loss', save_best_only=True)]

    training_acc = 0
    training_loss = 0
    valid_acc = 0
    valid_loss = 0
    iterations = 0
    
    avg_acc = 0
    avg_loss = 0
    avg_iterations = 0

    skf = KFold(n_splits=kfolds)

    for train_index, val_index in skf.split(X, y):

        print("Training on numSplit:",numSplits)
        numSplits += 1
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        model.load_weights(weightsDir + 'model_init.h5')
        history = model.fit(X_train,y_train,
                              callbacks = cbacks,
                              epochs=max_epochs,
                              batch_size=batch_size,
                              validation_data=(X_val,y_val), 
                              verbose = 0)

        #save the metrics for the best epoch, or the last one
        if(len(history.history['accuracy']) == max_epochs):
            iterations += max_epochs
            training_acc += history.history['accuracy'][max_epochs-1]
            training_loss += history.history['loss'][max_epochs-1]
            valid_acc += history.history['val_accuracy'][max_epochs-1]
            valid_loss += history.history['val_loss'][max_epochs-1]
        else:
            iterations += len(history.history['accuracy']) - patience_count
            i = len(history.history['accuracy']) - patience_count - 1
            training_acc += history.history['accuracy'][i]
            training_loss += history.history['loss'][i]
            valid_acc += history.history['val_accuracy'][i]
            valid_loss += history.history['val_loss'][i]
           
        	
    training_acc /= numSplits
    training_loss /= numSplits
    valid_acc /= numSplits
    valid_loss /= numSplits
    iterations /= numSplits*1.0

    avg_acc = valid_acc
    avg_loss = valid_loss
    avg_iterations = iterations
    
    # Return the average acc and loss and iterations (on the validation sample!)
    return avg_acc,avg_loss, avg_iterations


# Parameters
filters_list = [32, 64, 128]  
layers_list = [1,2,5,10]
optimizer = ['adam','adadelta']
acc_map = nested_defaultdict(float,3)
loss_map = nested_defaultdict(float,3)

# Grid Search
for iLayer, layers in enumerate(layers_list):
    for iFilter, filters in enumerate(filters_list):
        for iOpt, opt in enumerate(optimizer):
    
            print("Training:")
            print("Layers:",layers)
            print("Filters:", filters)
            print("Optimizer:", opt)
            
            #run train data through the network
            avg_acc,avg_loss,avg_iterations = kfold_network(x_train, y_train,n_kfolds,layers,filters,opt)
            
            #store and output results
            acc_map[iOpt][iLayer][iFilter] = avg_acc
            loss_map[iOpt][iLayer][iFilter] = avg_loss
            
            print(avg_acc, avg_loss, avg_iterations)
            print()

            json.dump(acc_map, open(plotDir+'acc_map.json', 'w'))
            json.dump(loss_map, open(plotDir+'loss_map.json', 'w'))


for iOpt, opt in enumerate(optimizer):
    plot_grid(acc_map[iOpt], "Filters", "Layers",
        filters_list, layers_list, 'Acc Grid Search with ' + opt, plotDir+'acc_'+opt+'.png')
    plot_grid(loss_map[iOpt], "Filters", "Layers",
        filters_list, layers_list, 'Loss Grid Search with ' + opt, plotDir+'loss_'+opt+'.png')


