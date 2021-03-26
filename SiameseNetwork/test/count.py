import os, sys
import numpy as np
import pickle as pkl

dataDir = '/store/user/llavezzo/disappearingTracks/siameseData/MC_muons/'

savedEvents = [None, None, None, None]
nDesiredPerClass = [1000, 100, 100, 1000]

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".root.npz" not in file): continue 

	print(file)

	fin = np.load(dataDir+file, allow_pickle=True)

	for i, iClass in enumerate(['class' + str(i) for i in range(4)]):

		eventsThisClassThisFile = fin[iClass]

		if len(eventsThisClassThisFile) == 0: continue

		if savedEvents[i] is None: savedEvents[i] = eventsThisClassThisFile
		elif len(savedEvents[i]) <= nDesiredPerClass[i]: savedEvents[i] = np.vstack((savedEvents[i], eventsThisClassThisFile))
	
	filled = 0
	for i in range(4):
		if savedEvents[i] is None: continue
		if len(savedEvents[i]) >= nDesiredPerClass[i]: filled+=1
	if filled == 4: break

np.save("test.npy",savedEvents)