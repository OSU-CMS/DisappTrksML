import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepAE import DeepAE

model_file = 'ae_train/model.h5'
arch = DeepAE()
arch.load_model(model_file)

preds = None
for i in range(100,120):
	events = np.load('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons/images_'+str(i)+'.root.npz',allow_pickle=True)['background']
	events = np.reshape(events,(len(events),100,4))
	if preds is None: preds = arch.model.predict(events)
	else: preds = np.concatenate((preds,arch.model.predict(events)))
np.savez_compressed("ae_preds.npy", events=events, preds=preds)