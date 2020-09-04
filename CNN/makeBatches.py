import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import VGG19
import json
import random
import sys
import pickle
import datetime
import getopt

import utils

dataDir = '/store/user/mcarrigan/disappearingTracks/AMSB/selected_600_1000_step3_tanh/'
outputDir = '/home/mcarrigan/scratch0/disTracksML/DisappTrksML/CNN/AMSB/'

batch_size = 256

# import count dicts
#with open(dataDir+'eCounts.pkl', 'rb') as f:
#	eCounts = pickle.load(f)
with open(dataDir+'bkgCounts.pkl', 'rb') as f:
	bkgCounts = pickle.load(f)

# fill lists of all events and files
b_events, b_files = [], []
for file, nEvents in bkgCounts.items():
	for evt in range(nEvents):
		b_events.append(evt)
		b_files.append(file)
#e_events, e_files = [], []
#for file, nEvents in eCounts.items():
#	for evt in range(nEvents):
#		e_events.append(evt)
#		e_files.append(file)

availableBkg = sum(list(bkgCounts.values()))
nBatches = int(availableBkg / batch_size)

bkgPerBatch = np.asarray([batch_size]*nBatches)
bkgPerBatch = bkgPerBatch.astype(int)

print(availableBkg, nBatches)
# make batches
bkg_event_batches, bkg_file_batches = utils.make_batches(b_events, b_files, bkgPerBatch, nBatches)
#e_event_batches, e_file_batches = utils.make_batches(e_events, e_files, ePerBatch, nBatches)
val_e_file_batches = [list(set([-1])) for _ in range(nBatches)]
val_e_event_batches = np.array([[0,0]]*nBatches)

print(bkg_file_batches)
print(bkg_event_batches)

np.save(outputDir + 'bkg_files_valBatches', bkg_file_batches)
np.save(outputDir + 'bkg_events_valBatches', bkg_event_batches)
np.save(outputDir + 'e_files_valBatches', val_e_file_batches)
np.save(outputDir + 'e_events_valBatches', val_e_event_batches)













