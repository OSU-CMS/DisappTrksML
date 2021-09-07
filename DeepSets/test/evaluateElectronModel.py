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
dirs = ["/store/user/llavezzo/disappearingTracks/electronsTesting/higgsino_700GeV_10000cm_fullSel_FIXED/"]
inputFiles = []
for d in dirs: inputFiles += glob.glob(d+'*.root.npz') 

totPreds = []
for i,fname in enumerate(inputFiles):
	print(i)

	skip, preds = arch.evaluate_npy(fname, obj=['signal', 'signal_infos'])
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	totPreds = np.append(totPreds,preds[:,1])

print cm
np.save("h10090_fullSel_preds.npy", totPreds)