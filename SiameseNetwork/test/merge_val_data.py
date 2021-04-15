import os, sys
import numpy as np
import pickle as pkl

dataDirs = [
	'/store/user/llavezzo/disappearingTracks/siameseData/SingleMuon_val/'
	]
savedEvents = [None]*len(dataDirs)
nDesiredPerClass = [1e8]*len(dataDirs)

for i, dataDir in enumerate(dataDirs):

	print dataDir

	for file in os.listdir(dataDir):

		if(".root.npz" not in file): continue 

		fin = np.load(dataDir+file, allow_pickle=True)
		if i == 0: label = 'sets'
		else: label = 'signal'
		eventsThisFile = fin[label]
		eventsThisFile = np.reshape(eventsThisFile, (len(eventsThisFile), 20, 7))

		if len(eventsThisFile) == 0: continue

		if savedEvents[i] is None: 
			savedEvents[i] = eventsThisFile
		elif len(savedEvents[i]) <= nDesiredPerClass[i]: 
			savedEvents[i] = np.vstack((savedEvents[i], eventsThisFile))

		if len(savedEvents[i]) >= nDesiredPerClass[i]: break

# savedEvents[0] = savedEvents[0][30:55]
# savedEvents[0] = savedEvents[0][:30]

np.save("singleMu_forTraining.npy", savedEvents)