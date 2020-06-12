import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import callbacks
from keras.metrics import FalseNegatives
from keras import backend as K
import numpy as np
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


dataDir = '/data/disappearingTracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn_gs/'
weightsDir = workDir + 'weights/cnn_gs/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

############config parameters###############
kfolds = 5
batch_size = 128
num_classes = 2
epochs = 100
patience_count = 20
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
############################################

# data, classes, and reco results
tag = '_DYJets50_norm_40x40'
data_e = np.load(dataDir+'e'+tag+'.npy')
data_m = np.load(dataDir+'muon'+tag+'.npy')
data_bkg = np.load(dataDir+'bkg'+tag+'.npy')
e_reco_results = np.load(dataDir + 'e_reco'+tag+'.npy')
m_reco_results = np.load(dataDir + 'muon_reco'+tag+'.npy')
bkg_reco_results = np.load(dataDir + 'bkg_reco'+tag+'.npy')

data = np.concatenate([data_e,data_m,data_bkg])
data = data[:,:,:,[0,2,3]]
classes = np.concatenate([np.ones(len(data_e)),np.zeros(len(data_m)),np.zeros(len(data_bkg))])
reco_results = np.concatenate([e_reco_results,m_reco_results,bkg_reco_results])

# select out events where the electron RECO failed
indices = [i for i,reco in enumerate(reco_results) if reco[0] > 0.15]
x = data[indices]
y = classes[indices]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def build_model(layers=1,filters=64,kernels=(1,1),opt='adam'):
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
    model.add(Dense(num_classes, activation='softmax',bias_initializer=output_bias))


def kfold_network(X, y, kfolds):
    
    numSplit = 0
    
    model = build_model()
    
    #early stopping
    patienceCount = patience_count
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience_count),
                 ModelCheckpoint(filepath=weightsDir+str(numSplits)+'_nlayers'+str(n_layers)+'_nhidden'+str(hidden_nodes)+'.h5', monitor='val_loss', save_best_only=True)]

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
        numSplit += 1
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        network.load_weights(weightsDir + 'model_init.h5')
        history = network.fit(X_train,y_train,
                              callbacks = callbacks,
                              epochs=max_epochs,
                              batch_size=batch_size,
                              validation_data=(X_val,y_val), 
                              verbose = 0)

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


parameters_list = []
acc_list = []
loss_list = []
iterations_list = []
nodes_list = [32, 64, 128]  
layers_list = [1,2,5,10]
optimizer = ['adam','adadelta']
acc_map = nested_defaultdict(double,2)
loss_map = nested_defaultdict(double,2)

# Determine best number of hidden nodes for one charge, and apply it for other charges
for iLayer, layers in enumerate(layers_list):
    for iNode, nodes in enumerate(nodes_list):
        for iOpt, opt in enumerate(optimizer)
    
            print("Training:")
            print("Layers:",layers)
            print("Nodes:", nodes)
            print("Optimizer:", opt)
            
            #run train data through the network
            avg_acc,avg_loss,avg_iterations = kfold_network(X_train, y_train,kfolds,layers,nodes,opt)
            
            #store and output results
            acc_map[i][j] = avg_acc
            loss_map[i][j] = avg_loss
            
            print(avg_mae, avg_loss, avg_iterations)


#FIXME: PLOT ACC, LOSS MAP


