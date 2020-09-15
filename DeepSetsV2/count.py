import os
import numpy as np
import sys
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee/"

eCounts = {}
bkgCounts = {}
electrons, bkg = 0, 0

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".npz" not in file): continue 

	index = file.find(".")
	fileNum = int(file[11:index])

	temp1 = np.load(dataDir+file)['e']
	np.save(dataDir+"e_0p5_"+str(fileNum)+".npy",temp1)
	eCounts.update({fileNum:temp1.shape[0]})
	electrons += temp1.shape[0]

	temp2 = np.load(dataDir+file)['bkg']
	np.save(dataDir+"bkg_0p5_"+str(fileNum)+".npy",temp2)
	bkgCounts.update({fileNum:temp2.shape[0]})
	bkg += temp2.shape[0]
	
with open(dataDir+'eCounts.pkl', 'wb') as f:
	pkl.dump(eCounts,f)
with open(dataDir+'bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)