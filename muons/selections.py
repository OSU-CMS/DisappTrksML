import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# print("Predicting reco-failed muons")
# events = np.load('muons.npy.npz',allow_pickle=True)['sets']
# events = np.reshape(events,(len(events),100,4))[:,:20,:]
# for i in range(events.shape[-1]): events[:,:,i] = normalize(events[:,:,i])
# events = np.reshape(events,(len(events),20*4))
# preds = arch.model.predict(events)
# np.savez_compressed("ae_train_traditional_reco_failures/muon_failures_preds.npy", events=events, preds=preds)

# tot, empty = 0, 0
# eps = 0.00000001
# inputFiles = glob.glob('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/images_*.root.npz')
# inputIndices = np.array([f.split('images_')[-1][:-9] for f in inputFiles])
# for i in range(500):
# 	print(i)
# 	data = np.load('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_muons/images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)['background']
# 	tot += len(data)
# 	for event in data:
# 		if np.sum(abs(event)) < eps: 
# 			empty+=1
# print("Found",empty,"empty events out of",tot)

events, event_infos = None, None
inputFiles = glob.glob('/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons/images_*.root.npz')
for file in inputFiles:
	print(file)
	data = np.load(file,allow_pickle=True)
	sets = data['signal']
	if(len(sets) == 0): continue
	else:
		infos = data['signal_info']
		if(events is None): 
			events = sets
			event_infos = infos
		else: 
			events = np.vstack((events,sets))
			event_infos = np.vstack((event_infos,infos))

np.savez_compressed("truthMuons_recoFailed.npz",sets=events,infos=event_infos)
