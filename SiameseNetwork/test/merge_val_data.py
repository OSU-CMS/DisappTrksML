import os, sys
import numpy as np
import pickle as pkl

dataDir_signal = '/store/user/llavezzo/disappearingTracks/siameseData/higgsino_700GeV_100cm/'
dataDir_muons = '/store/user/llavezzo/disappearingTracks/SingleMuon_2017F_v7/'
savedEvents = [None, None]
nDesiredPerClass = [55, 25]

for i, dataDir in enumerate([dataDir_signal, dataDir_muons]):

	print dataDir

	for file in os.listdir(dataDir):

		if(".root.npz" not in file): continue 

		fin = np.load(dataDir+file, allow_pickle=True)
		if i==0: label = 'signal'
		else: label = 'sets'
		eventsThisFile = fin[label]

		if len(eventsThisFile) == 0: continue

		if savedEvents[i] is None: savedEvents[i] = eventsThisFile
		elif len(savedEvents[i]) <= nDesiredPerClass[i]: savedEvents[i] = np.vstack((savedEvents[i], eventsThisFile))

		if len(savedEvents[i]) >= nDesiredPerClass[i]: break

# savedEvents[0] = savedEvents[0][30:55]
savedEvents[0] = savedEvents[0][:30]
print len(savedEvents[0])
print len(savedEvents[1])

np.savez_compressed("ref_data.npz", class0=savedEvents[0], class1=savedEvents[1])