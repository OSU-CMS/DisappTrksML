import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *
from DisappTrksML.DeepSets.MuonModel import *

# initialize the model with the weights
fileDir = 'train_muons/'
model_file = 'model.h5'
model_params = {
	'max_hits' : 20,
	'track_info_indices' : [
							4, 			# nPV
							6, 			# dRMinBadEcalChannel
							8, 9, 		# track eta, phi
							11, 		# nLayersWithMeasurement
							14, 15 		# sum of ECAL, HCAL energy
							]
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
predMu_tracks, predMu_infos = None, None
predBkg_tracks, predBkg_infos = None, None
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

	if predMu_tracks is None:
		predMu_tracks = data['tracks'][preds[:,1] > 0.5]
		predMu_infos = data['infos'][preds[:,1] > 0.5]
	else:
		predMu_tracks = np.vstack((predMu_tracks, data['tracks'][preds[:,1] > 0.5]))
		predMu_infos = np.vstack((predMu_infos, data['infos'][preds[:,1] > 0.5]))
	if predBkg_tracks is None:
		predBkg_tracks = data['tracks'][preds[:,1] <= 0.5]
		predBkg_infos = data['infos'][preds[:,1] <= 0.5]
	else:
		predBkg_tracks = np.vstack((predBkg_tracks, data['tracks'][preds[:,1] <= 0.5]))
		predBkg_infos = np.vstack((predBkg_infos, data['infos'][preds[:,1] <= 0.5]))


print cm
np.savez_compressed("muonPreds.npz",
					predMu_tracks = predMu_tracks,
					predMu_infos = predMu_infos,
					predBkg_tracks = predBkg_tracks,
					predBkg_infos = predBkg_infos)