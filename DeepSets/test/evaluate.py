import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.MuonModel import *

# initialize the model with the weights
fileDir = 'train_2021-03-25_16.46.49/'
model_file = 'model.9.h5'
model_params = {
	'eta_range':1.0,
	'phi_range':1.0,
	'phi_layers':[128,64,32],
	'f_layers':[64,32],
	'max_hits' : 20,
	'track_info_indices' : [4, 6, 8, 9, 11, 14, 15]
}
arch = MuonModel(**model_params)
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# evaluate the model
count = 0
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/SingleMuon_2017F_v7/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')

for i,fname in enumerate(inputFiles):
	print(i)

	data = np.load(fname, allow_pickle=True)
	count += data['sets'].shape[0]

	skip, preds = arch.evaluate_npy(fname, calos=False, obj='sets')
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	assert np.sum(cm) == count 			# had some issues with this with the two headed network
	print cm