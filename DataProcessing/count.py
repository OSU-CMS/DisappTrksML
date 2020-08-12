import os
import numpy as np
import sys
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/electron_selection/"
signal = 'e'
tag = '0p25_'

sCounts = {}
bkgCounts = {}

for fileNum in range(4000):

	f1 = signal+'_'+tag+str(fileNum)+'.npy'
	f2 = 'bkg_'+tag+str(fileNum)+'.npy'

	if(os.path.isfile(dataDir+f1)): 
		temp1 = np.load(dataDir+f1)
		sCounts.update({fileNum:temp1.shape[0]})

	if(os.path.isfile(dataDir+f1)): 
		temp2 = np.load(dataDir+f2)
		bkgCounts.update({fileNum:temp2.shape[0]})
	print(fileNum)


with open(dataDir+signal+'Counts.pkl', 'wb') as f:
	pkl.dump(sCounts,f)
with open(dataDir+'bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)