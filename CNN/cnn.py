import os
import keras
import tensorflow
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


dataDir = '/data/disappearingTracks/e_reco_failed/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn/'
weightsDir = workDir + 'weights/cnn/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

#config parameters
pos_class = [1]
neg_class = [0,2]
batch_size = 2048
epochs = 100
patience_count = 10
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)
oversample_val = 0.1
undersample_val = 0.2
smote_val = -1

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
x = df.iloc[:,4:].to_numpy()
x = np.reshape(x, [x.shape[0],40,40,4])
x = x[:,:,:,[0,2,3]]
y = df['type'].to_numpy()
for i,label in enumerate(y):
    if label in pos_class: y[i] = 1
    if label in neg_class: y[i] = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

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

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.save_weights(weightsDir+'initial_weights.h5')

callbacks = [
    callbacks.EarlyStopping(patience=patience_count),
    callbacks.ModelCheckpoint(filepath=weightsDir+'model.{epoch:02d}.h5'),
]

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=callbacks,
          class_weight = class_weight)

model.save_weights(weightsDir + 'first_model.h5')

plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.title('Accuracy History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plotDir+'accuracy_history.png')
plt.clf()

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.title('Loss History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(plotDir+'loss_history.png')
plt.clf()

predictions = model.predict(x_test)

print()
print("Calculating and plotting confusion matrix")
cm = calc_cm(y_test,predictions)
plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
print()

print("Plotting ceratainty")
plot_certainty(y_test,predictions,plotDir+'certainty.png')
print()

precision, recall = calc_binary_metrics(cm)
print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",round(precision,3))
print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",round(recall),3)
auc = roc_auc_score(y_test,predictions)
print("AUC Score:",round(auc,5))
print()