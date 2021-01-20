import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AE import AE


# initialize the model with the weights
fileDir = 'ae_genmuons_bkg/'
model_file = 'model.4.h5'
arch = AE()
arch.load_model(fileDir+model_file)

# predict muon reco failures
muons = np.load('truthMuons_recoFailed.npz',allow_pickle=True)['sets']
muons = np.reshape(muons[:,:20,:],(len(muons),20*4))
muon_preds = arch.model.predict(muons)
np.savez_compressed(fileDir+"muon_failures_preds.npy", events=muons, preds=muon_preds)

# predicting reconstructed muons
bkg_preds = None
bkg_events = None
bkgDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons_bkg/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(700,750):
	print(i)
	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)['background']

	data = np.divide(data + np.array([0.25,0.25,0,200]),np.array([0.5,0.5,4,1200]))
	assert len(data[np.where(data < 0)]) == 0, data[np.where(data < 0)]
	assert len(data[np.where(data > 1)]) == 0, data[np.where(data > 1)]

	data = np.reshape(data[:,:20,:],(len(data),20*4))
	if bkg_events is None: bkg_events = data
	else: bkg_events = np.concatenate((bkg_events,data))
	if bkg_preds is None: bkg_preds = arch.model.predict(bkg_events)
	else: bkg_preds = np.concatenate((bkg_preds,arch.model.predict(data)))
	assert bkg_preds.shape == bkg_events.shape
np.savez_compressed(fileDir + "background_preds.npy", events=bkg_events, preds=bkg_preds)

sys.exit(0)
def calc_losses(events,preds):
    losses = []
    for event, pred in zip(events, preds):
        loss = np.sum(abs(event-pred))
        losses.append(loss)
    return losses

losses =  calc_losses(muons, muon_preds)
val_losses = calc_losses(bkg_events, bkg_preds)

plt.hist(losses,bins=100,density=True,alpha=0.5,label="Muons - Reco. failures")
plt.hist(val_losses,bins=100,density=True,alpha=0.5,label="Background")
plt.legend()
plt.xlabel("Loss")
plt.yscale('log')
plt.savefig("preds.png")