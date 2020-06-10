import os
import keras
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import callbacks
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from collections import Counter
from utils import *

# dataDir = 'c:/users/llave/Documents/CMS/data/'
# workDir = 'c:/users/llave/Documents/CMS/'
dataDir = '/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
genDir = workDir + 'gan_gen/'
plotDir = workDir + 'plots/cnn_aug/'
weightsDir = workDir + 'weights/cnn_aug/'

os.system('mkdir '+str(plotDir))
os.system('mkdir '+str(weightsDir))

#config parameters
batch_size = 128
num_classes = 2
epochs = 100
patienceCount = 10  
genf = 'genImages.npy'

# input image dimensions
img_rows, img_cols = 40, 40
channels = 4
input_shape = (img_rows,img_cols,channels)

# load the data, split between train and test sets
data_e = np.load(dataDir+'e_DYJets50_v4_norm_40x40.npy')
data_bkg = np.load(dataDir+'bkg_DYJets50_v4_norm_40x40.npy')
e_reco_results = np.load(dataDir + 'e_reco_DYJets50_v4_norm_40x40.npy')
bkg_reco_results = np.load(dataDir + 'bkg_reco_DYJets50_v4_norm_40x40.npy')
classes = np.concatenate([np.ones(len(data_e)),np.zeros(len(data_bkg))])
data = np.concatenate([data_e,data_bkg])
data = data[:,:,:,1:]
reco_results = np.concatenate([e_reco_results,bkg_reco_results])

# load generated electron images
x_train_gen = np.load(genDir+genf)
y_train_gen = np.ones(len(x_train_gen))
print("Imported",len(x_train_gen),"GAN generated electron events")

# scale data
temp = np.concatenate([data,x_train_gen])
x_min = temp.min(axis=(1, 2, 3), keepdims=True) #FIXME: check if this is right
x_max = temp.max(axis=(1, 2, 3), keepdims=True)
temp = (temp - x_min)/(x_max-x_min)
data_scaled = temp[:len(data)]
x_train_gen_scaled = temp[len(data):]
assert len(data) == len(data_scaled) and len(x_train_gen_scaled) == len(x_train_gen), "error in scaling"

# split real data only
x_train, x_test, y_train, y_test, reco_train, reco_test = train_test_split(data_scaled, classes, reco_results, test_size=0.30, random_state=42)

# comine real and gen data, shuffle
x_train = np.concatenate([x_train,x_train_gen_scaled])
y_train = np.concatenate([y_train,y_train_gen])
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
assert len(x_train) == len(y_train), "Imported data error, len(x_train) != len(y_train)"

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# initialize output bias
neg, pos = np.bincount(y_train)
output_bias = np.log(pos/neg)
output_bias = keras.initializers.Constant(output_bias)
print("neg",neg,"pos",pos)

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
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization()) # testing
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization()) # testing
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',bias_initializer=output_bias))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


callbacks = [
    callbacks.EarlyStopping(monitor='val_loss',patience=patienceCount),
    callbacks.ModelCheckpoint(filepath=weightsDir+'best_model.{epoch:02d}.h5',save_best_only=True),
]

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=callbacks,
          class_weight = class_weight)

model.save_weights(weightsDir + 'first_gen_model.h5')

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

m = np.zeros([2,2,2])
for true,pred,reco in zip(y_test, predictions, reco_test):
    t = np.argmax(true)
    p = np.argmax(pred)
    m[t][p][reco] += 1
    
label = ['bkg','e']
print("Pred:\t\t\t bkg\t\t\t e")
for i in range(2):
    print("True: ",label[i],end="")
    for j in range(2):
        print("\t\t",int(m[i][j][0]),"\t",int(m[i][j][1]),end="")
    print()
print()