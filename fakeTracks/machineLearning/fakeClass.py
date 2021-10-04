import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

class fakeNN(keras.Model):

    def __init__(self, filters, input_dim, batch_norm, val_metrics):
        super(fakeNN, self).__init__()
        self.filters = filters
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        self.val_metrics = val_metrics

    def __call__(self):
        model = self.buildModel(self.filters, self.input_dim, self.batch_norm, self.val_metrics)
        return model

    def buildModel(self, filters, input_dim, batch_norm, val_metrics):
        model = Sequential()
        model.add(Dense(filters[0], input_dim=input_dim, activation='relu'))
        for i in range(len(filters)-1):
            model.add(Dense(filters[i+1], activation='relu'))
            if(batch_norm): model.add(BatchNormalization())
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=val_metrics)

        print(model.summary())
        return model
