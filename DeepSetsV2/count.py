import os
import numpy as np
import sys
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee_failAllRecos/"

tag = "_0p5"

eCounts = {}
bkgCounts = {}
electrons, bkg = 0, 0

for file in os.listdir(dataDir):

	if("images"+tag not in file): continue 
	if(".npz" not in file): continue 

	index = file.find(".")
	fileNum = file[11:index]
	print(fileNum)

	temp1 = np.load(dataDir+file)['e']
	eCounts.update({fileNum:temp1.shape[0]})
	electrons += temp1.shape[0]

	temp2 = np.load(dataDir+file)['bkg']
	bkgCounts.update({fileNum:temp2.shape[0]})
	bkg += temp2.shape[0]

print(electrons)
print(bkg)
with open(dataDir+'eCounts.pkl', 'wb') as f:
	pkl.dump(eCounts,f)
with open(dataDir+'bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)