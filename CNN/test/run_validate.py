import numpy as np
import validate
import utils
import tensorflow as tf
from tensorflow import keras
import json
from keras.models import load_model
import pickle

input_shape = (40,40,3)
#oversample_e = -1
#undersample_bkg = 0.5
#nTotE = 100
filters = [64, 128, 64]
batch_norm = False
batch_size = 256
metrics = ['Precision', 'Recall',
            'TruePositives','TrueNegatives',
            'FalsePositives', 'FalseNegatives']

dataDir = "/store/user/mcarrigan/disappearingTracks/convertedImages_higgsino_800GeV_1000cm_step3/"
workDir = ''
weightsDir = '/data/users/mcarrigan/cnn_9_11_reduced/cnn_9_11_reduced_p0/weights/'
outputDir = workDir + 'AMSB/higgsino_800_1000/'
plotDir = workDir + 'AMSB/higgsino_800_1000/' 

# import count dicts
with open(dataDir+'eCounts.pkl', 'rb') as f:
    eCounts = pickle.load(f)
with open(dataDir+'bCounts.pkl', 'rb') as f:
    bkgCounts = pickle.load(f)


def build_model(input_shape=(40,40,3), batch_norm = False, filters=[64,128,64],
                                output_bias=0, metrics=['accuracy']):

        model = keras.Sequential()

        model.add(keras.layers.Conv2D(filters[0], kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        if(batch_norm): model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))

        for layer in range(1,len(filters)):
                model.add(keras.layers.Conv2D(filters[layer], (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                if(batch_norm): model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Dense(1, activation='sigmoid',bias_initializer=keras.initializers.Constant(output_bias)))
        model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer="adam",
              metrics=metrics)
        print(model.summary())

        return model


model = build_model(input_shape = (40,40,3),
                            filters = filters, batch_norm=batch_norm,
                            #output_bias=output_bias, 
                            metrics=metrics)


model.load_weights(weightsDir+'lastEpoch.h5')

validate.run_validation(model, weightsDir+'lastEpoch.h5', outputDir, dataDir, plotDir)

