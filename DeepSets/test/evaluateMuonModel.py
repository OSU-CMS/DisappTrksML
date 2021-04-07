import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.MuonModel import *

# initialize the model with the weights
fileDir = 'training_output/'
model_file = 'model.h5'
model_params = {
	'max_hits' : 20,
	'track_info_indices' : [4, 6, 8, 9, 11, 14, 15]
}
arch = MuonModel(**model_params)
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# evaluate the model
count = 0
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/SingleMuon_pt1/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')

bkgDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/SingleMuon_pt2/"
inputFiles = inputFiles + glob.glob(bkgDir+'images_*.root.npz')

totPreds = []
for i,fname in enumerate(inputFiles):
	print(i)

	data = np.load(fname, allow_pickle=True)
	count += data['tracks'].shape[0]

	skip, preds = arch.evaluate_npy(fname, obj=['tracks', 'infos'])
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	totPreds = np.concatenate((totPreds, preds[:,1]))

	assert np.sum(cm) == count 	

print cm
np.savez_compressed("muonModel_TPMuons_trainSel.npz", signal_preds = totPreds)