import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import callbacks
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

# dataDir = 'c:/users/llave/Documents/CMS/data/'
# workDir = 'c:/users/llave/Documents/CMS/'
dataDir = '/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/cnn_smote/'
weightsDir = workDir + 'weights/cnn/'

#config parameters
num_classes = 2

# input image dimensions
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)

# the data, split between train and test sets
data = np.load(dataDir+'singleElectron2017_v4_norm_40x40.npy')
test_data = data[:,:,:,[1,3,4]]
reco_results = np.load(dataDir+'singleElectron2017_reco_v4_norm_40x40.npy')

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

model.load_weights(weightsDir + 'first_model.h5')

predictions = model.predict(test_data)

for pred, true in zip(predictions, reco_results):
    if(true == False):
        print(pred)
    if pred[0] > 0.9:
        pred = 0
    else:
        pred = 1
    if(true == False):
        print(pred)


# print()
# print("Calculating and plotting confusion matrix")
# cm = calc_cm(y_test,predictions)
# plot_confusion_matrix(cm,['bkg','e'],plotDir + 'cm.png')
# print()

# print("Plotting ceratainty")
# plot_certainty(y_test,predictions,plotDir+'certainty.png')
# print()

# precision, recall = calc_binary_metrics(cm)
# print("Precision = TP/(TP+FP) = fraction of predicted true actually true ",precision)
# print("Recall = TP/(TP+FN) = fraction of true class predicted to be true ",recall)
# auc = roc_auc_score(y_test,predictions)
# print("AUC Score:",auc)
# print()

# m = np.zeros([2,2,2])
# for true,pred,reco in zip(y_test, predictions, reco_test):
#     t = np.argmax(true)
#     p = np.argmax(pred)
#     m[t][p][reco] += 1
    
# label = ['bkg','e']
# print("Pred:\t\t\t bkg\t\t\t e")
# for i in range(2):
#     print("True: ",label[i],end="")
#     for j in range(2):
#         print("\t\t",int(m[i][j][0]),"\t",int(m[i][j][1]),end="")
#     print()
# print()