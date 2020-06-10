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
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from utils import *
import random
from sklearn.metrics import roc_auc_score

dataDir = '/data/disappearingTracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn_weights/'
weightsDir = workDir + 'weights/cnn_weights/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

#config parameters
batch_size = 128
num_classes = 2
epochs = 100
patience_count = 20
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)

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

#SMOTE and under sampling
# counter = Counter(y_train)
# print("Before",counter)
# x_train = np.reshape(x_train,[x_train.shape[0],40*40*5])
# oversample = SMOTE(sampling_strategy=0.5)
# undersample = RandomUnderSampler(sampling_strategy=0.75)
# steps = [('o', oversample), ('u', undersample)]
# pipeline = Pipeline(steps=steps)
# x_train, y_train = pipeline.fit_resample(x_train, y_train)
# counter = Counter(y_train)
# print("After",counter)
# x_train = np.reshape(x_train,[x_train.shape[0],40,40,5])

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

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',bias_initializer=output_bias))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

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
plt.legend()
plt.savefig(plotDir+'accuracy_history.png')
plt.clf()

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
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
print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",precision)
print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",recall)
auc = roc_auc_score(y_test,predictions)
print("AUC Score:",auc)
print()