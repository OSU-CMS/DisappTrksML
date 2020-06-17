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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from utils import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(40,40,3),
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

dataDir = '/data/disappearingTracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn/'
weightsDir = workDir + 'weights/cnn/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

#config parameters
fname = 'images_DYJets50_norm_40x40.pkl'
pos_class = [1]
neg_class = [0,2]
batch_size = 128
epochs = 100
patience_count = 20
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)

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

#SMOTE and under sampling
counter = Counter(y_train)
print("Before",counter)
x_train = np.reshape(x_train,[x_train.shape[0],40*40*3])
oversample = SMOTE(sampling_strategy=0.2)
undersample = RandomUnderSampler(sampling_strategy=0.3)
steps = [('o', oversample), ('u', undersample)]
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
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax',bias_initializer=output_bias))

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