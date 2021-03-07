import os, sys
import numpy as np
import pickle as pkl

dataDir = "/store/user/llavezzo/disappearingTracks/SingleMu_2017F_wIso_dR1p0/"

eCounts = {}
bkgCounts = {}
signal, bkg = 0, 0

infos = None
muons = None

for file in os.listdir(dataDir):

	if("images" not in file): continue 
	if(".root.npz" not in file): continue 

	print(file)

	index1 = file.find("_")
	index2 = file.find(".root.npz")
	fileNum = int(file[index1+1:index2])

	fin = np.load(dataDir+file, allow_pickle=True)

	signal_thisTree = int(fin['signal'].shape[0])
	# bkg_thisTree = int(fin['background'].shape[0])

	eCounts.update({fileNum:signal_thisTree})
	signal += signal_thisTree

	# bkgCounts.update({fileNum:bkg_thisTree})
	# bkg += bkg_thisTree

	if signal_thisTree == 0: continue
	infos_thisTree = fin['signal_infos']
	signal_thisTree = fin['signal']
	if infos is None: infos = infos_thisTree
	else: infos = np.vstack((infos, infos_thisTree))
	if muons is None: muons = signal_thisTree
	else: muons = np.vstack((muons, signal_thisTree))

print("signal",signal)
print("bkg",bkg)

np.savez_compressed("nonRecoProbeMuons.npz",signal=muons,signal_infos=infos)

# with open(dataDir+'sCounts.pkl', 'wb') as f:
# 	pkl.dump(eCounts,f)
# with open(dataDir+'bkgCounts.pkl', 'wb') as f:
# 	pkl.dump(bkgCounts,f)
