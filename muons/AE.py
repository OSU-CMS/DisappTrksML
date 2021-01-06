#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pickle

from ROOT import TFile, TTree

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization, concatenate, Reshape
from tensorflow.keras import optimizers, regularizers, callbacks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AE:

    model = None
    training_history = None

    def __init__(self, eta_range=0.25, phi_range=0.25, maxHits=100,
        phi_layers = [128, 64], f_layers = [64, 128], track_info_shape = 0):
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.maxHits = maxHits

        self.track_info_shape = track_info_shape

        self.phi_layers = phi_layers
        self.f_layers = f_layers

    def buildModel(self):

        inputs = Input(shape=(self.maxHits*4))
        normed = BatchNormalization()(inputs)
        encoded = Dense(128, activation='relu')(normed)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(self.maxHits*4, activation='sigmoid')(decoded)

        autoencoder = keras.Model(inputs, decoded)

        self.model = autoencoder

        print(self.model.summary())

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def load_model_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def evaluate_model(self, event, track):
        event = self.convertTrackFromTreeElectrons(event, track, 1) # class_label doesn't matter
        #prediction = self.model.predict(np.reshape(event['sets'], (1, 100, 4)))
        prediction = self.model.predict([np.reshape(event['sets'], (1, 100, 4)),np.reshape(event['infos'],(1,13))[:,[4,8,9]]])
        return prediction[:,1] # p(is electron)

    # def evaluate_model(self, event, track):
    #     converted_arrays = self.convertTrackFromTreeMuons(event, track, 1) # class_label doesn't matter
    #     prediction = self.model.predict([np.reshape(converted_arrays['sets'], (1, 100, 4)),np.reshape(converted_arrays['infos'],(1,13))[:,[8,9,10,11]]])
    #     return prediction[:,1] # p(is electron)

    def fit_generator(self, train_generator, val_generator=None, epochs=10, monitor='val_loss',patience_count=3,outdir=""):
        self.model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        training_callbacks = [
            callbacks.EarlyStopping(monitor=monitor,patience=patience_count),
            callbacks.ModelCheckpoint(filepath=outdir+'model.{epoch}.h5',
                                            save_best_only=True,
                                            monitor=monitor,
                                            mode='auto')
        ]

        self.training_history = self.model.fit(train_generator, 
                                             validation_data=val_generator,
                                             callbacks=training_callbacks,
                                             epochs=epochs,
                                             verbose=1)

    def save_model(self, outputFileName):
        self.model.save(outputFileName)
        print 'Saved model in file:', outputFileName

    def save_weights(self, outputFileName):
        self.model.save_weights(outputFileName)
        print 'Saved model weights in file:', outputFileName

    def save_trainingHistory(self, outputFileName):
        with open(outputFileName, 'wb') as f:
            pickle.dump(self.training_history.history, f)
        print 'Saved training history in file:', outputFileName

    def displayTrainingHistory(self):
        acc = self.training_history.history['accuracy']
        val_acc = self.training_history.history['val_accuracy']

        loss = self.training_history.history['loss']
        val_loss = self.training_history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    def plotHistory(self,infile,outfile,metric='loss'):

        with open(infile,'rb') as f:
            history = pickle.load(f)

        loss = history[metric]
        val_loss = history['val_'+metric]

        epochs = range(1, len(loss) + 1)

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training '+metric)
        plt.plot(epochs, val_loss, 'b', label='Validation '+metric)
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(outfile)