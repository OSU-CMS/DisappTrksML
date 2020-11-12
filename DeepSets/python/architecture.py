#!/usr/bin/env python

import math
from datetime import datetime
import numpy as np
import pickle

from ROOT import TFile, TTree

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers

import matplotlib.pyplot as plt

# combine EB+EE and muon detectors into ECAL/HCAL/MUO indices
def detectorIndex(detType):
    if detType == 1 or detType == 2:
        return 0
    elif detType == 4:
        return 1
    elif detType >= 5 and detType <= 7:
        return 2
    else:
        return -1

# return (dEta, dPhi) between track and hit
def imageCoordinates(track, hit):
    dEta = track.eta - hit.eta
    dPhi = track.phi - hit.phi
    # branch cut [-pi, pi)
    if abs(dPhi) > math.pi:
        dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
    return (dEta, dPhi)

def isGenMatched(event, track, pdgId):
    matchID = 0
    matchDR2 = -1
    for p in event.genParticles:
        if p.pt < 10:
            continue
        if not p.isPromptFinalState and not p.isDirectPromptTauDecayProductFinalState:
            continue

        dEta = track.eta - p.eta
        dPhi = track.phi - p.phi
        if abs(dPhi) > math.pi:
            dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
        dR2 = dEta*dEta + dPhi*dPhi

        if matchDR2 < 0 or dR2 < matchDR2:
            matchDR2 = dR2
            matchID = p.pdgId

    return (abs(matchID) == pdgId and abs(matchDR2) < 0.1**2)

class DeepSetsArchitecture:
    eta_range = 0.25
    phi_range = 0.25
    max_hits = 100

    # model parameters
    phi_layers = [64, 64, 256]
    f_layers = [64, 64, 64]

    model = None
    training_history = None

    def __init__(self, eta_range=0.25, phi_range=0.25, maxHits=100):
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.max_hits = maxHits

        self.input_shape = (self.max_hits, 4)

    def set_phi_layers(self, layers):
        self.phi_layers = layers

    def set_f_layers(self, layers):
        self.f_layers = layers

    def convertTrackFromTree(self, event, track, class_label):
        hits = []

        for hit in event.recHits:
            dEta, dPhi = imageCoordinates(track, hit)
            if abs(dEta) >= self.eta_range or abs(dPhi) >= self.phi_range:
                continue
            detIndex = detectorIndex(hit.detType)
            energy = hit.energy if detIndex != 2 else 1
            hits.append((dEta, dPhi, energy, detIndex))

        if len(hits) > 0:
            hits = np.reshape(hits, (len(hits), 4))
            hits = hits[hits[:, 2].argsort()]
            hits = np.flip(hits, axis=0)
            assert np.max(hits[:, 2]) == hits[0, 2]

        sets = np.zeros(self.input_shape)
        for i in range(min(len(hits), self.max_hits)):
            for j in range(4):
                sets[i][j] = hits[i][j]

        infos = np.array([event.eventNumber, event.lumiBlockNumber, event.runNumber,
                          class_label,
                          event.nPV,
                          track.deltaRToClosestElectron,
                          track.deltaRToClosestMuon,
                          track.deltaRToClosestTauHad,
                          track.eta,
                          track.phi,
                          track.dRMinBadEcalChannel,
                          track.nLayersWithMeasurement,
                          track.nValidPixelHits])

        values = {
            'sets' : sets,
            'infos' : infos,
        }

        return values

    def eventSelection(self, event):
        trackPasses = []
        for track in event.tracks:
            if (abs(track.eta) >= 2.4 or
                track.inGap or
                abs(track.dRMinJet) < 0.5 or
                abs(track.deltaRToClosestElectron) < 0.15 or
                abs(track.deltaRToClosestMuon) < 0.15 or
                abs(track.deltaRToClosestTauHad) < 0.15):
                trackPasses.append(False)
            else:
                trackPasses.append(True)
        return (True in trackPasses), trackPasses

    def convertFileToNumpy(self, fileName):
        inputFile = TFile(fileName, 'read')
        inputTree = inputFile.Get('trackImageProducer/tree')

        signal = []
        signal_info = []
        background = []
        background_info = []

        for event in inputTree:
            eventPasses, trackPasses = self.eventSelection(event)
            if not eventPasses: continue

            for i, track in enumerate(event.tracks):
                if not trackPasses[i]: continue

                if isGenMatched(event, track, 11):
                    values = self.convertTrackFromTree(event, track, 1)
                    signal.append(values['sets'])
                    signal_info.append(values['infos'])
                else:
                    values = self.convertTrackFromTree(event, track, 0)
                    background.append(values['sets'])
                    background_info.append(values['infos'])

        outputFileName = fileName.split('/')[-1] + '.npz'

        np.savez_compressed(outputFileName,
                            signal=signal,
                            signal_info=signal_info,
                            background=background,
                            background_info=background_info)

        print 'Wrote', outputFileName

        inputFile.Close()

    def buildModel(self):
        inputs = Input(shape=(self.input_shape[-1],))

        # build phi network for each individual hit
        phi_network = Masking()(inputs)
        for layerSize in self.phi_layers[:-1]:
            phi_network = Dense(layerSize)(phi_network)
            phi_network = Activation('relu')(phi_network)
            phi_network = BatchNormalization()(phi_network)
        phi_network = Dense(self.phi_layers[-1])(phi_network)
        phi_network = Activation('linear')(phi_network)

        # build summed model for latent space
        unsummed_model = Model(inputs=inputs, outputs=phi_network)
        set_input = Input(shape=self.input_shape)
        phi_set = TimeDistributed(unsummed_model)(set_input)
        summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
        phi_model = Model(inputs=set_input, outputs=summed)

        # define F (rho) network evaluating in the latent space
        f_inputs = Input(shape=(self.phi_layers[-1],)) # plus any other track/event-wide variable
        f_network = Dense(self.f_layers[0])(f_inputs)
        f_network = Activation('relu')(f_network)
        for layerSize in self.f_layers[1:]:
            f_network = Dense(layerSize)(f_network)
            f_network = Activation('relu')(f_network)
        f_network = Dense(2)(f_network)
        f_outputs = Activation('softmax')(f_network)
        f_model = Model(inputs=f_inputs, outputs=f_outputs)

        # build the DeepSets architecture
        deepset_inputs = Input(shape=self.input_shape)
        latent_space = phi_model(deepset_inputs)
        deepset_outputs = f_model(latent_space)
        model = Model(inputs=deepset_inputs, outputs=deepset_outputs)

        print(model.summary())

        self.model = model

    def load_model_weights(self, weights_path):
        self.buildModel()
        self.model.load_weights(weights_path)

    def evaluate_model(self, track):
        return self.model.predict(track)

    def fit_generator(self, train_generator, validation_data, epochs=10):
        self.model.compile(optimizer=optimizers.Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.training_history = self.model.fit_generator(train_generator,
                                                         validation_data=validation_data,
                                                         epochs=epochs)

        backup_suffix = datetime.now().strftime('%Y-%M-%d_%H.%M.%S')
        self.save_weights('model_' + backup_suffix + '.h5')
        pickle.dump(self.training_history, open('trainingHistory_' + backup_suffix + '.pkl', 'wb'))

    def save_weights(self, outputFileName):
        self.model.save_weights(outputFileName)
        print 'Saved models in file:', outputFileName

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
