import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepSetsMuons import *

# initialize the model with the weights
fileDir = 'train_genMuons_bkg/'
model_file = 'model.20.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

count = 0
# evaluate the model
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/images_higgsino_700GeV_10cm_fullSelection_withCalos/"
inputFiles = glob.glob(bkgDir+'hist_*.root.npz')
# inputFiles = ['../nonRecoProbeMuons.npz']

for i,fname in enumerate(inputFiles):
	print(i)
	#if i == 100: break

	data = np.load(fname, allow_pickle=True)
	count += data['signal'].shape[0]

	skip, preds = arch.evaluate_npy(fname, calos=False, 
									info_indices=[4, 6, 8, 9, 10, 11, 12, 13, 14, 15],
									obj='signal')
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	assert np.sum(cm) == count 			# had some issues with this with the two headed network
	print cm