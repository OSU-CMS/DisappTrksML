import numpy as np
import tensorflow as tf
from tensorflow import keras
from AE import AE
import glob, os, sys
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_file = 'ae_train_traditional_reco_failures/model.h5'
arch = AE()
arch.load_model(model_file)

# arch.plotHistory('ae_train/trainingHistory.pkl','ae_train/trainingHistory.png','loss')

print("Predicting reco-failed muons")
events = np.load('muons.npy.npz',allow_pickle=True)['sets']
events = np.reshape(events,(len(events),100,4))[:,:20,:]
for i in range(events.shape[-1]): events[:,:,i] = normalize(events[:,:,i])
events = np.reshape(events,(len(events),20*4))
preds = arch.model.predict(events)
np.savez_compressed("ae_train_traditional_reco_failures/muon_failures_preds.npy", events=events, preds=preds)

print("Predicting reconstructed muons")
bkg_preds = None
bkg_events = None
inputFiles = glob.glob('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/images_*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
for i in range(700,750):
	data = np.load('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)['background']
	data = np.reshape(data,(len(data),100,4))[:,:20,:]
	for i in range(data.shape[-1]): data[:,:,i] = normalize(data[:,:,i])
	data = np.reshape(data,(len(data),20*4))
	if bkg_events is None: bkg_events = data
	else: bkg_events = np.concatenate((bkg_events,data))
	if bkg_preds is None: bkg_preds = arch.model.predict(bkg_events)
	else: bkg_preds = np.concatenate((bkg_preds,arch.model.predict(data)))
	assert bkg_preds.shape == bkg_events.shape
np.savez_compressed("ae_train_traditional_reco_failures/background_preds.npy", events=bkg_events, preds=bkg_preds)

def calc_losses(events,preds):
    losses = []
    for event, pred in zip(events, preds):
        loss = np.sum(abs(event-pred))
        losses.append(loss)
    return losses

print(bkg_events.shape)
print(bkg_preds.shape)
losses =  calc_losses(events, preds)
val_losses = calc_losses(bkg_events, bkg_preds)

print(len(val_losses))
print(val_losses[:100])

plt.hist(losses,bins=100,density=True,alpha=0.5,label="Reconstructed Muons")
plt.hist(val_losses,bins=100,density=True,alpha=0.5,label="Unreconstructed Muons")
plt.legend()
plt.xlabel("Loss")
plt.yscale('log')
plt.savefig("preds.png")