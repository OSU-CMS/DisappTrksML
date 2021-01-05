import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepAE import DeepAE

model_file = 'train_2021-01-05_14.27.34/model.h5'
arch = DeepAE()
arch.load_model(model_file)

events = np.load('muons.npy.npz',allow_pickle=True)['sets']
events = np.reshape(events,(len(events),100,4))
preds = arch.model.predict(events)
np.savez_compressed("ae_preds_muons.npy", events=events, preds=preds)