import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, auc
import sys
from itertools import product 
import cnn
import utils


dataDir = '/store/user/llavezzo/images/'
tag = '_0p25_tanh'
workDir = '/data/users/llavezzo/cnn/'
plotDir = workDir + 'plots/'
weightsDir = workDir + 'weights/'


#########k-fold parameters####################################################
n_kfolds = 4

filters_list = [32, 64, 128]  
layers_list = [2,5,10]
optimizer = ['adam','adadelta']
parameters = list(product(layers_list, filters_list, optimizer))

parameterNum = int(sys.argv[1])
parameters = parameters[parameterNum]
n_layers = parameters[0]
n_filters = parameters[1]
opt = parameters[2]

print("Running on:",n_layers,"layers,",n_filters,"filters,",opt,"optimizer")
#############################################################################


###########cnn config parameters##################
pos_class = [1]
neg_class = [0,2]
batch_size = 2048
max_epochs = 30
patience_count = 10
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
oversample_val = 0.1
undersample_val = 0.2
##################################################


images, infos = utils.load_electron_data(dataDir, tag)

x = images[:,:-1]
x = np.reshape(x, [len(x),40,40,4])
x = x[:,:,:,[0,2,3]]

y = np.array([x[6] for x in infos])
y = y.astype(int)


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

    x_train, y_train = utils.apply_oversampling(x_train,y_train,oversample_val=oversample_val)
    x_train, y_train = utils.apply_undersampling(x_train,y_train,undersample_val=undersample_val)

    # initialize output bias
    neg, pos = np.bincount(y_train)
    output_bias = np.log(pos/neg)
    output_bias = keras.initializers.Constant(output_bias)
    print("Positive Class Counter:",pos)
    print("Negative Class Counter:",neg)

    model = cnn.build_model(input_shape = input_shape, layers = n_layers, filters = n_filters, opt=opt, output_bias=output_bias)
    
    weightsFile = 'numSplit'+str(numSplit)+'_params'+str(parameterNum)+'.h5'

    history = cnn.train_model(model,x_train,y_train,x_val,y_val,
                        weightsDir,weightsFile,
                        patience_count=patience_count,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        class_weights = True)

    model.load_weights(weightsDir+weightsFile)

    predictions = model.predict(x_val)
    cm = utils.calc_cm(y_val,predictions)
    precision, recall = utils.calc_binary_metrics(cm)
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
np.save('gsresults_'+str(sys.argv[1]),[avg_acc,avg_loss, avg_iterations, avg_precision, avg_recall, avg_auc])