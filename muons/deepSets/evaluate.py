import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepSetsMuons import *

# initialize the model with the weights
fileDir = 'trainV2/'
model_file = 'model.2.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# predicting reconstructed muons
# bkgDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons_bkg/"
# inputFiles = glob.glob(bkgDir+'images_*.root.npz')
# inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
# for i in range(600,700):
# 	print(i)
# 	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

# 	bkg = data['background'][:,:30]
# 	bkg_info = data['background_info'][:,[4,8,9,13,14]]

# 	bkg_preds = arch.model.predict([bkg,bkg_info])

# 	cm[0,1] += np.count_nonzero(bkg_preds[:,1] > 0.5)
# 	cm[0,0]+= np.count_nonzero(bkg_preds[:,1] <= 0.5)

# predicting non-reco muons
count = 0
bkgDir = "/store/user/llavezzo/disappearingTracks/nonRecoGenMuons_v6/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(len(inputFiles)):
	print(i)

	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

	count += data['signal'].shape[0]
	# events = data['signal'][:,:40]
	# infos = data['signal_info'][:,[4,8,9,13,14,15,16]]

	# preds = arch.model.predict([events,infos])

	# cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
	# cm[0,0]+= np.count_nonzero(preds[:,1] <= 0.5)

print cm
print count