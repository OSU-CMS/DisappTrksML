import os
import numpy as np
import sys
import pickle as pkl

dataDir = "/store/user/mcarrigan/disappearingTracks/electron_selection_DYJetsZtoEE_tanh/"

if len(sys.argv) > 1: dataDir = str(sys.argv[1])

signal = 'e'
tag = '0p25_'

sCounts = {}
bkgCounts = {}
electrons = 0
bkg = 0

for fileNum in range(4000):

	f1 = signal+'_'+tag+str(fileNum)+'.npz'
	f2 = 'bkg_'+tag+str(fileNum)+'.npz'

	if(os.path.isfile(dataDir+f1)): 
		temp1 = np.load(dataDir+f1)['infos']
		sCounts.update({fileNum:temp1.shape[0]})
		electrons += temp1.shape[0]

	if(os.path.isfile(dataDir+f1)): 
		temp2 = np.load(dataDir+f2)['infos']
		bkgCounts.update({fileNum:temp2.shape[0]})
		bkg+=temp2.shape[0]

print(electrons)
print(bkg)
with open(signal+'Counts.pkl', 'wb') as f:
	pkl.dump(sCounts,f)
with open('bkgCounts.pkl', 'wb') as f:
	pkl.dump(bkgCounts,f)