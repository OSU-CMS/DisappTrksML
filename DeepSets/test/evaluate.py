import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.ElectronModel import *

# initialize the model with the weights
fileDir = '/data/users/llavezzo/forBrian/kfold19_noBatchNorm_finalTrainV3/'
model_file = 'model.h5'
model_params = {
	'phi_layers':[400,256,128], 
	'f_layers': [128,128,64,32],
	'track_info_indices' : [4,8,9,12]
}
arch = ElectronModel(**model_params)
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# evaluate the model
count = 0
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/electronsTesting/SingleEle_fullSel_pt1_FIXED/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')

totPreds = []
for i,fname in enumerate(inputFiles):
	print(i)

	data = np.load(fname, allow_pickle=True)
	count += data['tracks'].shape[0]

	skip, preds = arch.evaluate_npy(fname, obj=['tracks', 'infos'])
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	assert np.sum(cm) == count 		
	# print cm
	totPreds = np.append(totPreds,preds[:,1])

print cm
np.save("SingleEle_fullSel_pt2_preds.npy", totPreds)