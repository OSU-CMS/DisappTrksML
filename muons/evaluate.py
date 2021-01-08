import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepAE import DeepAE
import glob, os, sys

model_file = 'ae_train/model.h5'
arch = DeepAE()
arch.load_model(model_file)

arch.plotHistory('ae_train/trainingHistory.pkl','ae_train/trainingHistory.png','loss')

print("Predicting reco-failed muons")
events = np.load('muons.npy.npz',allow_pickle=True)['sets']
events = np.reshape(events,(len(events),100,4))[:,:50,:]
preds = arch.model.predict(events)
np.savez_compressed("ae_train/ae_preds_muons.npy", events=events, preds=preds)

print("Predicting reconstructed muons")
preds = None
events = None
inputFiles = glob.glob('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons/images_*.root.npz')
inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
for i in range(250,300):
	data = np.load('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons/images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)['background']
	data = np.reshape(data,(len(data),100,4))[:,:50,:]
	if events is None: events = data
	else: events = np.concatenate((events,data))
	if preds is None: preds = arch.model.predict(events)
	else: preds = np.concatenate((preds,arch.model.predict(data)))
	assert preds.shape == events.shape
np.savez_compressed("ae_train/ae_preds.npy", events=events, preds=preds)