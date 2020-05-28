from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
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
import utils
import random
from sklearn.metrics import roc_auc_score

dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/'
weightsDir = workDir + 'weights/cnn_aug/'

#config parameters
batch_size = 64
num_classes = 2
epochs = 50

# input image dimensions
img_rows, img_cols = 40, 40
channels = 5
input_shape = (img_rows,img_cols,channels)

# the data, split between train and test sets
data_e = np.load(dataDir+'e_DYJets50V3_norm_40x40.npy')
data_bkg = np.load(dataDir+'bkg_DYJets50V3_norm_40x40.npy')
classes = np.concatenate([np.ones(len(data_e)),np.zeros(len(data_bkg))])
data = np.concatenate([data_e,data_bkg])
x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=0.30, random_state=42)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(x_train)

neg, pos = np.bincount(y_train)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

output_bias = np.log(pos/neg)
output_bias = keras.initializers.Constant(output_bias)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(40, 40,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1,bias_initializer=output_bias))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',      #FIXME: Adam?
              metrics=['accuracy','AUC','Precision','Recall'])

              
weight_for_0 = (1/neg)*(neg+pos)/2.0
weight_for_1 = (1/pos)*(neg+pos)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, 
          epochs=epochs,
          validation_data=(x_test, y_test),
          class_weight = class_weight)
          
model.save_weights(weightsDir+'first_try.h5')  # always save your weights after training or during training
