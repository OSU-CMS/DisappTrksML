import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepSetsMuons import *

# initialize the model with the weights
fileDir = 'trainV3/'
model_file = 'model.2.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

# predicting reconstructed muons
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/SingleMu_2017F_wIso/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(len(inputFiles)):
	print(i)
	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

	if(data['infos'].shape[1]  < 15): 
		print("fuck")
		continue
	bkg = data['sets'][:,:40]
	bkg_info = data['infos'][:,[4,8,9,13,14,15,16]]

	preds = arch.model.predict([bkg,bkg_info])

	cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
	cm[0,0]+= np.count_nonzero(preds[:,1] <= 0.5)

# # predicting non-reco muons
# sig_preds = None
# sigDir = "/store/user/llavezzo/disappearingTracks/nonRecoGenMuons_v6/"
# inputFiles = glob.glob(sigDir+'images_*.root.npz')
# inputIndices = np.array([f.split(sigDir+'images_')[-1][:-9] for f in inputFiles])
# for i in range(len(inputFiles)):

# 	data = np.load(sigDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

# 	events = data['signal'][:,:40]
# 	infos = data['signal_info'][:,[4,8,9,13,14,15,16]]

# 	preds = arch.model.predict([events,infos])

# 	cm[1,1] += np.count_nonzero(preds[:,1] > 0.5)
# 	cm[1,0]+= np.count_nonzero(preds[:,1] <= 0.5)

# 	if sig_preds is None: sig_preds = preds
# 	else: sig_preds = np.vstack((preds,sig_preds))

# preds = np.concatenate((bkg_preds[:,1],sig_preds[:,1]))
# true = np.concatenate((np.zeros(len(bkg_preds)), np.ones(len(sig_preds))))
# arch.plot_classifier_output(true, preds, fileDir+"predsV3.png")
print cm