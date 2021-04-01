import os, sys
import numpy as np
import pickle as pkl

dataDir = '/store/user/llavezzo/disappearingTracks/siameseData/MC_muons_V2/'

savedEvents = [None, None, None, None]
savedInfos = [None, None, None, None]
nDesiredPerClass = [100000, 100, 10000, 100000]

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".root.npz" not in file): continue 

	print(file)

	fin = np.load(dataDir+file, allow_pickle=True)

	for i, iClass in enumerate(['class' + str(i) for i in range(4)]):

		eventsThisClassThisFile = fin[iClass]
		infosThisClassThisFile = fin[iClass+"_infos"]

		if len(eventsThisClassThisFile) == 0: continue

		if savedEvents[i] is None: 
			savedEvents[i] = eventsThisClassThisFile
			savedInfos[i] = infosThisClassThisFile
 		elif len(savedEvents[i]) <= nDesiredPerClass[i]:
 			savedEvents[i] = np.vstack((savedEvents[i], eventsThisClassThisFile))
 			savedInfos[i] = np.vstack((savedInfos[i], infosThisClassThisFile))
	
	filled = 0
	for i in range(4):
		if savedEvents[i] is None: continue
		if len(savedEvents[i]) >= nDesiredPerClass[i]: filled+=1
		
	if filled == 4: break

# savedEvents2 = []
# savedEvents2.append(savedEvents[1])
# savedEvents2.append(savedEvents[2])
# for iClass in [0,3]:
# 	sorted_indices = np.argsort(savedInfos[iClass][:,15])
# 	sorted_indices = np.array_split(sorted_indices, 4)
# 	splitEvents = []
# 	splitInfos = []
# 	for indices in sorted_indices:
# 		splitEvents.append(savedEvents[iClass][indices])
# 		splitInfos.append(savedInfos[iClass][indices])
# 	savedEvents2 = np.concatenate((savedEvents2, splitEvents))
# print(savedEvents2.shape)

np.savez_compressed("train_data.npz", 
					class0=savedEvents[0], class1=savedEvents[1],
					class2=savedEvents[2], class3=savedEvents[3],
					class0_infos=savedInfos[0], class1_infos=savedInfos[1],
					class2_infos=savedInfos[2], class3_infos=savedInfos[3] )