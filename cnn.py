import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/'
weightsDir = workDir + 'weights/cnn/'

#config parameters
batch_size = 64
num_classes = 2
epochs = 300

# input image dimensions
img_rows, img_cols = 20, 20
input_shape = (img_rows,img_cols,3)

# the data, split between train and test sets
data_e = np.load(dataDir+'e_DYJets50_norm_20x20.npy')
data_bkg = np.load(dataDir+'bkg_DYJets50_norm_20x20.npy')
classes = np.concatenate([np.ones(len(data_e)),np.zeros(len(data_bkg))])
data = np.concatenate([data_e,data_bkg])

#shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
classes = classes[indices]

split = int(data.shape[0]*0.7)
x_train = data[:split]
x_test = data[split:]
y_train = classes[:split]
y_test = classes[split:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.legend()
plt.savefig(plotDir+'accuracy_history.png')
plt.clf()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.savefig(plotDir+'loss_history.png')

predictions = model.predict(x_test)

from collections import defaultdict
from functools import partial
from itertools import repeat

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()

confusion_matrix = nested_defaultdict(int,2)

correct_certainty = []
notcorrect_certainty = []

for true,pred in zip(y_test, predictions):
    if np.argmax(true) == np.argmax(pred):
        confusion_matrix[np.argmax(true)][np.argmax(pred)] += 1
        correct_certainty.append(pred[np.argmax(pred)])
    else:
        confusion_matrix[np.argmax(true)][np.argmax(pred)] += 1
        notcorrect_certainty.append(pred[np.argmax(pred)])
        

label = ['bkg','e']
print()
print("Pred:\t\t bkg\t e")
for i in range(2):
    print("True: ",label[i],end="")
    for j in range(2):
        print("\t",confusion_matrix[i][j],end="")
    print()
print()


c1=1
c2=0
TP = confusion_matrix[c1][c1]
FP = confusion_matrix[c2][c1]
FN = confusion_matrix[c1][c2]
TN = confusion_matrix[c2][c2]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",precision)
print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",recall)
print()