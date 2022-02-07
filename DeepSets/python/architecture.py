#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import math
from datetime import datetime
import numpy as np
import pickle

from ROOT import TFile, TTree

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization, concatenate
from tensorflow.keras import optimizers, regularizers, callbacks

import matplotlib
matplotlib.use('Agg')
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

def printEventInfo(event, track):
    print('EVENT INFO')
    print('Trigger:', event.firesGrandOrTrigger)
    print('MET filters:', event.passMETFilters)
    print('Num good PVs (>=1):', event.numGoodPVs)
    print('MET (no mu) (>120):', event.metNoMu)
    print('Num good jets (>=1):', event.numGoodJets)
    print('max dijet dPhi (<=2.5):', event.dijetDeltaPhiMax)
    print('dPhi(lead jet, met no mu) (>0.5):', abs(event.leadingJetMetPhi))
    print()
    print('TRACK INFO')
    print('\teta (<2.1):', abs(track.eta))
    print('\tpt (>55):', track.pt)
    print('\tIn gap (false):', track.inGap)
    print('\tNot in 2017 low eff. region (true):', (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42))
    print('\tmin dR(track, bad ecal channel) (>= 0.05):', track.dRMinBadEcalChannel)
    print('\tnValidPixelHits (>=4):', track.nValidPixelHits)
    print('\tnValidHits (>=4):', track.nValidHits)
    print('\tmissing inner hits (==0):', track.missingInnerHits)
    print('\tmissing middle hits (==0):', track.missingMiddleHits)
    print('\ttrackIso/pt (<0.05):', track.trackIso / track.pt)
    print('\td0 (<0.02):', abs(track.d0))
    print('\tdz (<0.5):', abs(track.dz))
    print('\tmin dR(track, jet) (>0.5):', abs(track.dRMinJet))
    print('\tmin dR(track, ele) (>0.15):', abs(track.deltaRToClosestElectron))
    print('\tmin dR(track, muon) (>0.15):', abs(track.deltaRToClosestMuon))
    print('\tmin dR(track, tauHad) (>0.15):', abs(track.deltaRToClosestTauHad))
    print('\tecalo (<10):', track.ecalo)
    print('\tmissing outer hits (>=3):', track.missingOuterHits)
    print()
    print('\tisTagProbeElectron:', track.isTagProbeElectron)
    print('\tisTagProbeMuon:', track.isTagProbeMuon)

class DeepSetsArchitecture:

    model = None
    training_history = None

    def __init__(self, eta_range=0.25, phi_range=0.25, max_hits=100, phi_layers=[64, 64, 256], f_layers=[64, 64, 64], track_info_indices=0):
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.max_hits = max_hits

        self.input_shape = (self.max_hits, 4)
        self.track_info_indices = track_info_indices

        self.phi_layers = phi_layers
        self.f_layers = f_layers

    def eventSelectionSignal(self, event):
        eventPasses = (event.firesGrandOrTrigger == 1 and
            event.passMETFilters == 1 and
            event.numGoodPVs >= 1 and
            event.metNoMu > 120 and
            event.numGoodJets >= 1 and
            event.dijetDeltaPhiMax <= 2.5 and
            abs(event.leadingJetMetPhi) > 0.5)

        trackPasses = [False] * len(event.tracks)

        if not eventPasses:
            return eventPasses, trackPasses

        for i, track in enumerate(event.tracks):
            if (not abs(track.eta) < 2.1 or
                not track.pt > 55 or
                track.inGap or
                not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
                continue

            if (not track.dRMinBadEcalChannel >= 0.05 or
                not track.nValidPixelHits >= 4 or
                not track.nValidHits >= 4 or
                not track.missingInnerHits == 0 or
                not track.missingMiddleHits == 0 or
                not track.trackIso / track.pt < 0.05 or
                not abs(track.d0) < 0.02 or
                not abs(track.dz) < 0.5 or
                not abs(track.dRMinJet) > 0.5 or
                not abs(track.deltaRToClosestElectron) > 0.15 or
                not abs(track.deltaRToClosestMuon) > 0.15 or
                not abs(track.deltaRToClosestTauHad) > 0.15 or
                not track.ecalo < 10 or
                not track.missingOuterHits >= 3):
                continue

            trackPasses[i] = True

        return (True in trackPasses), trackPasses

    def eventSelectionFakeBackground(self, event):
        eventPasses = (event.passMETFilters == 1)
        trackPasses = [False] * len(event.tracks)

        if not eventPasses:
            return eventPasses, trackPasses

        for i, track in enumerate(event.tracks):
            if (not abs(track.eta) < 2.1 or
                not track.pt > 30 or
                track.inGap or
                not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
                continue

            if (not track.dRMinBadEcalChannel >= 0.05 or
                not track.nValidPixelHits >= 4 or
                not track.nValidHits >= 4 or
                not track.missingInnerHits == 0 or
                not track.missingMiddleHits == 0 or
                not track.trackIso / track.pt < 0.05 or
                # d0 sideband
                not abs(track.d0) >= 0.05 or
                not abs(track.d0) < 0.5 or
                not abs(track.dz) < 0.5 or
                not abs(track.dRMinJet) > 0.5 or
                not abs(track.deltaRToClosestElectron) > 0.15 or
                not abs(track.deltaRToClosestMuon) > 0.15 or
                not abs(track.deltaRToClosestTauHad) > 0.15 or
                not track.ecalo < 10 or
                not track.missingOuterHits >= 3):
                continue

            trackPasses[i] = True

        return (True in trackPasses), trackPasses

    def eventSelectionLeptonBackground(self, event, lepton_type):

        eventPasses = (event.passMETFilters == 1)
        trackPasses = [False] * len(event.tracks)
        trackPassesVeto = [False] * len(event.tracks)

        if not eventPasses:
            return eventPasses, trackPasses, trackPassesVeto

        for i, track in enumerate(event.tracks):
            if (not abs(track.eta) < 2.1 or
                not track.pt > 30 or
                track.inGap or
                not (track.phi < 2.7 or track.eta < 0 or track.eta > 1.42)): # 2017 eta-phi low efficiency
                continue

            if (lepton_type == 'electrons' and not track.isTagProbeElectron == 1):
                continue
            if (lepton_type == 'muons' and not track.isTagProbeMuon == 1):
                continue

            if (not track.dRMinBadEcalChannel >= 0.05 or
                not track.nValidPixelHits >= 4 or
                not track.nValidHits >= 4 or
                not track.missingInnerHits == 0 or
                not track.missingMiddleHits == 0 or
                not track.trackIso / track.pt < 0.05 or
                not abs(track.d0) < 0.02 or
                not abs(track.dz) < 0.5 or
                not abs(track.dRMinJet) > 0.5 or
                not abs(track.deltaRToClosestTauHad) > 0.15):
                continue

            if (lepton_type == 'electrons' and not abs(track.deltaRToClosestMuon) > 0.15):
                continue
            if (lepton_type == 'muons' and (not abs(track.deltaRToClosestElectron) > 0.15 or not track.ecalo < 10)):
                continue

            trackPasses[i] = True

            if lepton_type == 'electrons':
                if (abs(track.deltaRToClosestElectron) > 0.15 and
                    track.ecalo < 10 and
                    track.missingOuterHits >= 3):
                    trackPassesVeto[i] = True
            if lepton_type == 'muons':
                if (abs(track.deltaRToClosestMuon) > 0.15 and
                    track.missingOuterHits >= 3):
                    trackPassesVeto[i] = True

        return (True in trackPasses), trackPasses, trackPassesVeto

    def load_model(self, model_path):
        try:
            self.model = keras.models.load_model(model_path)
        except:
            self.model = keras.models.load_model(model_path, custom_objects={'tf': tf})


    def load_model_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def fit_generator(self, train_generator, val_generator=None, 
                      epochs=10, monitor='val_loss',patience_count=10,
                      metrics = ['accuracy'],
                      outdir=""):

        self.model.compile(optimizer=optimizers.Adagrad(), loss='categorical_crossentropy', metrics=metrics)
        
        training_callbacks = [
            callbacks.EarlyStopping(monitor=monitor, patience=patience_count),
            # callbacks.ModelCheckpoint(filepath=outdir + 'model.{epoch}.h5',
            #                           save_best_only=True,
            #                           monitor=monitor,
            #                           mode='auto')
        ]

        if val_generator is None:
            self.training_history = self.model.fit(train_generator,
                                                    epochs=epochs,
                                                    verbose=2)

        else:
            self.training_history = self.model.fit(train_generator, 
                                                   validation_data=val_generator,
                                                   callbacks=training_callbacks,
                                                   epochs=epochs,
                                                   verbose=2)
        
    def save_model(self, outputFileName):
        self.model.save(outputFileName)
        print('Saved model in file:', outputFileName)

    def save_weights(self, outputFileName):
        self.model.save_weights(outputFileName)
        print('Saved model weights in file:', outputFileName)

    def save_trainingHistory(self, outputFileName):
        with open(outputFileName, 'wb') as f:
            pickle.dump(self.training_history.history, f)
        print('Saved training history in file:', outputFileName)

    def displayTrainingHistory(self):
        acc = self.training_history.history['accuracy']
        val_acc = self.training_history.history['val_accuracy']

        loss = self.training_history.history['loss']
        val_loss = self.training_history.history['val_loss']

        epochs = list(range(1, len(acc) + 1))

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

    def plot_trainingHistory(self,infile,outfile,metric='loss'):
        with open(infile, 'rb') as f:
            history = pickle.load(f)

        loss = history[metric]
        val_loss = history['val_'+metric]

        epochs = list(range(1, len(loss) + 1))

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training '+metric)
        plt.plot(epochs, val_loss, 'b', label='Validation '+metric)
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(outfile)

    def save_metrics(self, infile, outfile, train_params):
        file = open(infile, 'rb')
        history = pickle.load(file)
        if(len(history['val_loss']) == train_params['epochs']):
            val_loss = history['val_loss'][-1]
            val_acc = history['val_accuracy'][-1]
        else:
            i = len(history['val_loss']) - train_params['patience_count'] - 1
            val_loss = history['val_loss'][i]
            val_acc = history['val_accuracy'][i]
        file.close()
        metrics = {
            "val_loss":val_loss,
            "val_acc":val_acc
        }
        with open(outfile, 'wb') as f:
            pickle.dump(metrics, f)

    def save_kfoldMetrics(self, infiles, outfile, train_params):
        k = len(infiles)
        val_loss, val_acc = 0,0
        for infile in infiles:
            file = open(infile,'rb')
            history = pickle.load(file)
            if(len(history['val_loss']) == train_params['epochs']):
                val_loss += history['val_loss'][-1]
                val_acc += history['val_accuracy'][-1]
            else:
                i = len(history['val_loss']) - train_params['patience_count'] - 1
                val_loss += history['val_loss'][i]
                val_acc += history['val_accuracy'][i]
            file.close()

        val_loss /= k
        val_acc /= k
        metrics = {
            "val_loss":val_loss,
            "val_accuracy":val_acc
        }

        with open(outfile, 'wb') as f:
            pickle.dump(metrics, f)

    # expects [c1,c1] = TP, [c2,c2] = TN
    def calc_binary_metrics(self, confusion_matrix, c1=1, c2=0):
        TP = confusion_matrix[c1][c1]
        FP = confusion_matrix[c2][c1]
        FN = confusion_matrix[c1][c2]
        TN = confusion_matrix[c2][c2]

        if((TP+FP) == 0): precision = 0
        else: precision = TP / (TP + FP)
        if((TP+FN) == 0): recall = 0
        else: recall = TP / (TP + FN)

        f1 = TP / (TP + 0.5*(FP + FN))

        return precision, recall, f1

    def calc_cm(self, true, predictions, dim=2):
        confusion_matrix = np.zeros((dim, dim))
        for t,p in zip(true.astype(int), predictions.astype(int)):
            confusion_matrix[t,p] += 1
        return confusion_matrix

    # plot precision and recall for different classifier outputs
    def metrics_per_cut(self, true, preds, nsplits=20):
        precisions, recalls, f1s, splits = [],[],[], []

        for split in np.arange(0,1,1.0/nsplits):
            class_labels = np.zeros(len(preds),dtype='int')
            class_labels[np.where(preds > split)] = 1
   
            cm = self.calc_cm(true,class_labels)

            precision, recall, f1 = self.calc_binary_metrics(cm)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            splits.append(split)

        metrics = {
            'splits':splits,
            'precision':precisions,
            'recall':recalls,
            'f1':f1s
        }

        return metrics
