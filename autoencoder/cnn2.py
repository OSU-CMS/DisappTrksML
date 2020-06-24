import os
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import callbacks
from tensorflow.keras import regularizers
from keras.metrics import FalseNegatives
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import json
from utils import *
import sklearn


imgDir = '/data/disappearingTracks/cleaned/images/'
classDir = '/data/disappearingTracks/cleaned/classes/'
workDir = '/home/mcarrigan/disTracksML/'
plotDir = workDir + 'plots/cnn/'
weightsDir = workDir + 'weights/cnn/'

#os.system('mkdir '+str(plotDir))
#os.system('mkdir '+str(weightsDir))

#config parameters
pos_class = [1]
neg_class = [0,2]
batch_size = 32
epochs = 100
patience_count = 10
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
oversample_val = 0.1
undersample_val = 0.2
smote_val = -1

total_events = 2000
file_batches = 2
fb_size = total_events/file_batches

file_count = 0
file_num = []
unused_events = []
cm = np.zeros((2,2))
pred = []
real = []

for filename in os.listdir(imgDir):
    if 'images' and 'Clean' in filename: 
        file_count += 1
        file_num.append(filename)
        this_file = np.load(imgDir + filename)
        this_events = np.arange(this_file.shape[0])
        unused_events.append(this_events)

#load data and classes


for fb in range(file_batches):
    print("Loading " + str(fb_size) + " events")
    this_batch, classes, unused_events = getBatch(fb_size, file_num, unused_events, imgDir, classDir)
    classes = classes[:, 0]
    classes = classes.flatten()
    classes = classes.astype('int64')
    for i in range(classes.shape[0]):
        if classes[i] == 1: classes[i] = 1
        if classes[i] != 1: classes[i] = 0

    x_train, x_test, y_train, y_test = train_test_split(this_batch, classes, test_size=0.20, random_state=42)
    
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(y_train.shape, 'classes')

    #SMOTE, over sampling, and under sampling
    counter = Counter(y_train)
    print("Before",counter)
    x_train = np.reshape(x_train,[x_train.shape[0],40*40*3])
    steps = []
    if(smote_val != -1): 
        print("Applying SMOTE with value",smote_val)
        smote = SMOTE(sampling_strategy=smote_val)
        steps.append(('o',smote))
    if(oversample_val != -1): 
        print("Applying oversampling with value",oversample_val)
        oversample = RandomOverSampler(sampling_strategy=oversample_val)
        steps.append(('o',oversample))
    if(undersample_val != -1): 
        print("Applying undersampling with value",undersample_val)
        undersample = RandomUnderSampler(sampling_strategy=undersample_val)
        steps.append(('u', undersample))
    pipeline = Pipeline(steps=steps)
    #x_train, y_train = pipeline.fit_resample(x_train, y_train)
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

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax',bias_initializer=output_bias))

    if (fb != 0): model.load_weights(weightsDir + 'model_' + str(fb-1) + '.h5')

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    
    Callbacks = [callbacks.EarlyStopping(patience=patience_count),
        callbacks.ModelCheckpoint(filepath=weightsDir+'model_' + str(fb) + '.h5'),
    ]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test), callbacks=Callbacks, class_weight = class_weight)

    model.save_weights(weightsDir + 'model_' + str(fb) + '.h5')
    #print("file count: ", file_count)
    print(y_test[:5])
    predictions = model.predict(x_test)
   
    for k in range(len(predictions)):
        pred.append(np.argmax(predictions[k]))
        real.append(np.argmax(y_test[k]))
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(real, pred).flatten()
    print("Confusion Matrix: ", tn, fp, fn, tp)
    cm[0][0] += tn
    cm[0][1] += fp
    cm[1][0] += fn
    cm[1][1] += tp 


plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.title('Accuracy History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plotDir+'accuracy_history_v1.png')
plt.clf()

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.title('Loss History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plotDir+'loss_history_v1.png')
plt.clf()

#predictions = model.predict(x_test)
    
print()
print("Calculating and plotting confusion matrix")
cm = cm.astype(float)
target_names = ['Background', 'Electron']
plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.tight_layout()
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(plotDir + 'confusion_matrix.png', bbox_inches='tight')
plt.clf()
print()

#print("Plotting ceratainty")
#plot_certainty(real,pred,plotDir+'certainty_v1.png')
#print()

#precision, recall = calc_binary_metrics(cm)
precision = sklearn.metrics.precision_score(real, pred)
recall = sklearn.metrics.recall_score(real, pred)
print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall,3))
auc = roc_auc_score(real,pred)
print("AUC Score:",round(auc,5))
print()
#    file_count += 1

