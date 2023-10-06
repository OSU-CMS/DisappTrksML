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

    def __init__(self, filters, input_dim, batch_norm, val_metrics, dropout, learning_rate, batch_size):
        super(fakeNN, self).__init__()
        self.filters = filters
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        self.val_metrics = val_metrics
        self.dropout = dropout
        self.dense = keras.layers.Dense(1, activation='sigmoid', name='Output')
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def __call__(self):
        model = self.buildModel(self.filters, self.input_dim, self.batch_norm, self.val_metrics, self.dropout, self.learning_rate, self.batch_size)
        return model

    def call(self, inputs):
        return self.dense(inputs)

    def getter(self):
        model = self.buildModelV2(self.filters, self.input_dim, self.batch_norm, self.val_metrics, self.dropout, self.learning_rate, self.batch_size)
        return model

    def buildModel(self, filters, input_dim, batch_norm, val_metrics, dropout, learning_rate, batch_size):
        model = Sequential()
        model.add(Dense(filters[0], input_dim=input_dim, activation='relu', name="Input"))
        for i in range(len(filters)-1):
            model.add(Dense(filters[i+1], activation='relu'))
            if(batch_norm): model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid', name="Output_xyz"))

        opt = keras.optimizers.Adam(lr=learning_rate)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=val_metrics)

        print(model.summary())
        return model

    def buildModelV2(self, filters, input_dim, batch_norm, val_metrics, dropout, learning_rate, batch_size):
        model = Sequential()
        model.add(Dense(filters[0], input_shape=(input_dim,), activation='relu', name="Input"))

        for i in range(len(filters)-1):
            model.add(Dense(filters[i+1], activation='relu'))
            if(batch_norm): model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid', name="Output_xyz"))

        return model