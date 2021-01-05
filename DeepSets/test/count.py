import os, sys
import numpy as np
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_genmuons/"

eCounts = {}
bkgCounts = {}
signal, bkg = 0, 0

signal_data = None

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".root.npz" not in file): continue 

	print(file)

	index1 = file.find("_")
	index2 = file.find(".root.npz")
	fileNum = int(file[index1+1:index2])

	fin = np.load(dataDir+file, allow_pickle=True)

	signal_thisTree = int(fin['signal'].shape[0])
	bkg_thisTree = int(fin['background'].shape[0])

	eCounts.update({fileNum:signal_thisTree})
	signal += signal_thisTree

	bkgCounts.update({fileNum:bkg_thisTree})
	bkg += bkg_thisTree
	
	if signal_thisTree > 0:
		if signal_data is None: 
			signal_data = fin['signal']
			signal_infos = fin['signal_info']
		else: 
			signal_data = np.vstack((signal_data,fin['signal']))
			signal_infos = np.vstack((signal_infos,fin['signal_info']))

print("signal",signal)
print("bkg",bkg)

np.savez_compressed("muons.npy",sets=signal_data,infos=signal_infos)
# with open(dataDir+'sCounts.pkl', 'wb') as f:
# 	pkl.dump(eCounts,f)
# with open(dataDir+'bkgCounts.pkl', 'wb') as f:
# 	pkl.dump(bkgCounts,f)
