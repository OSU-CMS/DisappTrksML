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
bkgDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/SingleMuon_fullSel_pt2_FIXED/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
# bkgDir = "/store/user/llavezzo/disappearingTracks/muonsTesting/SingleMuon_pt2/"
# inputFiles = inputFiles + glob.glob(bkgDir+'images_*.root.npz')
totPreds = []
empty = 0
for i,fname in enumerate(inputFiles):
	print(i)

	data = np.load(fname, allow_pickle=True)

	if(data['tracks'].shape[0] == 0): continue
	sets = data['tracks'][:, :20]
	infos = data['infos'][:, [4, 6, 8, 9, 11, 14, 15]]

	for event, info in zip(sets, infos):

		empty_event = False
		if np.sum(abs(event)) < 1e-8: 
			empty+=1
			empty_event = True

		x = [np.reshape(event, (1,20,7)), np.reshape(info, (1,7))]
		preds = arch.model.predict(x)

		if empty_event:
			cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
			cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)
		else:
			cm[1,1] += np.count_nonzero(preds[:,1] > 0.5)
			cm[1,0] += np.count_nonzero(preds[:,1] <= 0.5)

	totPreds = np.concatenate((totPreds, preds[:,1]))

print cm
# np.savez_compressed("muonModel_TPMuons_trainSel2.npz", signal_preds = totPreds)
print "empty", empty