import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, auc
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from utils import *
import random
import json
import sys
from itertools import product 


dataDir = '/store/user/llavezzo/e_reco_failed/'
workDir = '/data/users/llavezzo/cnn/'
plotDir = workDir + 'plots/cnn_gs/'
weightsDir = workDir + 'weights/cnn_gs/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

filters_list = [32, 64, 128]  
layers_list = [2,5,10]
optimizer = ['adam','adadelta']
parameters = list(product(layers_list, filters_list, optimizer))

parameter = int(sys.argv[1])
parameters = parameters[parameter]
n_layers = parameters[0]
n_filters = parameters[1]
opt = parameters[2]

###########config parameters##################
pos_class = [1]
neg_class = [0,2]
batch_size = 2048
max_epochs = 100
patience_count = 10
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
n_kfolds = 5
oversample_val = 0.1
undersample_val = 0.2
smote_val = -1
#############################################


# load partition
with open(dataDir+'partition.json') as json_file:
    partition = json.load(json_file)

# load data
df = pd.DataFrame()
for filename in os.listdir(dataDir):
    if "batch" in filename:
        i=filename.find("_")+1
        f=filename.find(".pkl")
        # skip test set
        if int(filename[i:f]) in partition['test']: continue
        dftemp = pd.read_pickle(dataDir+filename)
        df = pd.concat([df,dftemp])

# reshape data and separate classes
x = df.iloc[:,4:].values
x = np.reshape(x, [x.shape[0],40,40,4])
x = x[:,:,:,[0,2,3]]
y = df['type'].values
for i,label in enumerate(y):
    if label in pos_class: y[i] = 1
    if label in neg_class: y[i] = 0


def build_model(layers=1,filters=64,opt='adadelta',kernels=(1,1),output_bias=0):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    for _ in range(layers-1):
        model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax',bias_initializer=keras.initializers.Constant(0)))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

    return model


skf = KFold(n_splits=n_kfolds, shuffle = True)

numSplit = 0

# metrics
training_acc = 0
training_loss = 0
valid_acc = 0
valid_loss = 0
iterations = 0
valid_precision = 0
valid_recall = 0
valid_auc = 0

for train_index, val_index in skf.split(x, y):

    print("Training on numSplit:",numSplit)
    numSplit += 1
    x_train = x[train_index]
    y_train = y[train_index]
    x_val = x[val_index]
    y_val = y[val_index]

    #SMOTE, over sampling, and under sampling
    counter = Counter(y_train)
    print("Before",counter)
    x_train = np.reshape(x_train,[x_train.shape[0],40*40*3])
    steps = []
    oversample = RandomOverSampler(sampling_strategy=0.1)
    steps.append(('o',oversample))
    undersample = RandomUnderSampler(sampling_strategy=0.2)
    steps.append(('u', undersample))
    pipeline = Pipeline(steps=steps)
    x_train, y_train = pipeline.fit_resample(x_train, y_train)
    counter = Counter(y_train)
    print("After",counter)
    x_train = np.reshape(x_train,[x_train.shape[0],40,40,3])

    # initialize output bias
    neg, pos = np.bincount(y_train)
    output_bias = np.log(pos/neg)
    output_bias = keras.initializers.Constant(output_bias)
    print("Positive Class Counter:",pos)
    print("Negative Class Counter:",neg)

    # output weights
    weight_for_0 = (1/neg)*(neg+pos)/2.0
    weight_for_1 = (1/pos)*(neg+pos)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    model = build_model(n_layers, n_filters, opt, output_bias)
    model.save_weights(weightsDir + 'model_init.h5')

    y_train = keras.utils.to_categorical(y_train, 2)
    y_val = keras.utils.to_categorical(y_val, 2)

    # early stopping
    patienceCount = patience_count
    cbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_count),
                keras.callbacks.ModelCheckpoint(filepath=weightsDir+str(numSplit)+'_nlayers'+str(n_layers)+'_nfilters'+str(n_filters)+'_opt'+str(opt)+'.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(x_train,y_train,
                          callbacks = cbacks,
                          epochs=max_epochs,
                          batch_size=batch_size,
                          validation_data=(x_val,y_val), 
                          class_weight=class_weight,
                          verbose = 2)

    predictions = model.predict(x_val)
    cm = calc_cm(y_val,predictions)
    precision, recall = calc_binary_metrics(cm)
    auc = roc_auc_score(y_val,predictions)

    #save the metrics for the best epoch, or the last one
    if(len(history.history['acc']) == max_epochs):
        iterations += max_epochs
        training_acc += history.history['acc'][max_epochs-1]
        training_loss += history.history['loss'][max_epochs-1]
        valid_acc += history.history['val_acc'][max_epochs-1]
        valid_loss += history.history['val_loss'][max_epochs-1]
    else:
        iterations += len(history.history['acc']) - patience_count
        i = len(history.history['acc']) - patience_count - 1
        training_acc += history.history['acc'][i]
        training_loss += history.history['loss'][i]
        valid_acc += history.history['val_acc'][i]
        valid_loss += history.history['val_loss'][i]
    
    valid_precision += precision
    valid_recall += recall
    valid_auc += auc
       
    	
training_acc /= numSplit
training_loss /= numSplit
valid_acc /= numSplit
valid_loss /= numSplit
iterations /= numSplit*1.0
valid_precision /= numSplit
valid_recall /= numSplit
valid_auc /= numSplit

avg_acc = valid_acc
avg_loss = valid_loss
avg_iterations = iterations
avg_precision = valid_precision
avg_recall = valid_recall
avg_auc = valid_auc

# save the average acc and loss and iterations (on the validation sample!)
json.dump([avg_acc,avg_loss, avg_iterations, avg_precision, avg_recall, avg_auc], open('gs_results/gs_nlayers'+str(n_layers)+'_nfliters'+str(n_filters)+'_opt'+str(opt)+'.json', 'w'))