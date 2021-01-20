import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepSetsMuons import *

# initialize the model with the weights
fileDir = 'train/'
model_file = 'model.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# predicting reconstructed muons
bkgDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons_bkg/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(600,700):
	print(i)
	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

	bkg = data['background'][:,:30]
	bkg_info = data['background_info'][:,[4,8,9,13,14]]

	bkg_preds = arch.model.predict([bkg,bkg_info])

	cm[0,1] += np.count_nonzero(bkg_preds[:,1] > 0.5)
	cm[0,0]+= np.count_nonzero(bkg_preds[:,1] <= 0.5)

# predicting non-reco muons
data = np.load('truthMuons_recoFailed.npz',allow_pickle=True)
muons = data['sets'][:,:30]
muons_info = data['infos'][:,[4,8,9,13,14]]
for i in range(len(muons_info)):
	muons_info[i,3] = np.sum(muons_info[i,3])
	muons_info[i,4] = np.sum(muons_info[i,4])
muon_preds = arch.model.predict([muons,muons_info.astype(np.float64)])
cm[1,1] += np.count_nonzero(muon_preds[:,1] > 0.5)
cm[1,0]+= np.count_nonzero(muon_preds[:,1] <= 0.5)

print cm